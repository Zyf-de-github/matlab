clc; clear; close all;
addpath('./pcm');

%% ========= 第13组参数 =========
Rb = 64e3;                 % 比特率 
Fc = 10*Rb;                % 载波
fs = 12*Fc;                % 采样率（应满足 fs/Rb 为整数）
Ns = fs/Rb;                % 每比特采样点
assert(abs(Ns-round(Ns))<1e-12,'fs/Rb 必须为整数'); 
Ns = round(Ns);

%% ========= (A) 生成模拟输入：3 正弦叠加 =========
Fs_in = 48e3;                          % 生成"模拟信号"的采样率
Tsig  = 0.25;                          % 信号时长，秒（可调）
t_in  = (0:1/Fs_in:Tsig-1/Fs_in);

% --- 三路正弦 ---
f1=3; f2=10; f3=20;                    % 三个频率
a1=0.8; a2=0.5; a3=0.3;                % 不同幅度

sig1 = a1*sin(2*pi*f1*t_in);
sig2 = a2*sin(2*pi*f2*t_in);
sig3 = a3*sin(2*pi*f3*t_in);

% --- 合成并归一 ---
y_analog = sig1 + sig2 + sig3;
y_analog = y_analog/max(abs(y_analog))*0.98;   % 归一防溢

% --- 画三路 + 合成波形 ---
figure;
plot(t_in, sig1, 'r'); hold on;
plot(t_in, sig2, 'g');
plot(t_in, sig3, 'b');
plot(t_in, y_analog, 'w','LineWidth',1.2);
grid on;
legend('sin1','sin2','sin3','sum');
title('输入模拟信号（3正弦叠加 + 单独显示三路）');
xlabel('t/s'); ylabel('amp');

%% ========= (B) PCM 13 折线编码（8kHz） =========
sampleVal = Rb/8; % 8 kHz ⇒ 8 bit/样本 ⇒ 64 kbps，刚好匹配 Rb
[sampleData, bits_pcm] = PCM_13Encode(y_analog, Fs_in, sampleVal);

% PCM 抽样后波形
t_smp = (0:numel(sampleData)-1)/sampleVal;
figure; plot(t_smp, sampleData); grid on;
title('PCM 抽样后波形'); xlabel('t/s');

% PCM 编码 bit 流（截前 100 位展示）
figure; stairs(bits_pcm(1:100)); ylim([-0.2 1.2]); grid on;
title('PCM 编码后 bit（前 100 位）');

%% ========= (C.X) 码型变换：矩形脉冲 vs SRRC 脉冲（演示） =========

Nshow = 20;
bps = bits_pcm(1:Nshow);

% bit → ±1
symbols_demo = 2*bps - 1;

% SRRC 参数（演示 + 正式都用同一组）
beta = 0.75;          % 滚降系数
span = 8;             % 滤波器跨度（符号数）
srrc = rcosdesign(beta, span, Ns, 'sqrt');   % 根升余弦脉冲
delay_rrc = span * Ns;   % 两个 SRRC 级联总群时延（符号数*Ns）

% ========== 1) 矩形脉冲 ==========
rect = upsample(symbols_demo, Ns);
rect_wave = filter(ones(1, Ns), 1, rect);
% ========== 2) SRRC 脉冲 ==========
srrc_wave_demo = filter(srrc, 1, upsample(symbols_demo, Ns));

%% ========= (C.Y) 乘法器：基带 → 带通（演示） =========
t_show = (0:length(rect_wave)-1)/fs;
carrier_demo = cos(2*pi*Fc*t_show);

rect_mod = rect_wave .* carrier_demo;
srrc_mod_demo = srrc_wave_demo .* carrier_demo;

Nplot = min(2000, length(rect_wave));

figure;
subplot(2,2,1);
plot(rect_wave(1:Nplot),'b'); grid on;
title('矩形脉冲（基带）');
ylabel('幅度');

subplot(2,2,2);
plot(rect_mod(1:Nplot),'b'); grid on;
title('矩形脉冲 × cos(2πF_ct)（带通）');
ylim([-2 2]);

subplot(2,2,3);
plot(srrc_wave_demo(1:Nplot),'r'); grid on;
title('SRRC 脉冲（基带）');
xlabel('样本点'); ylabel('幅度');

subplot(2,2,4);
plot(srrc_mod_demo(1:Nplot),'r'); grid on;
title('SRRC 脉冲 × cos(2πF_ct)（带通）');
xlabel('样本点');

%% ========= (C.Z) 频谱：未成形 vs SRRC =========
Nfft = min([4096, length(rect_mod), length(srrc_mod_demo)]);

[fx_rect, S_rect]   = simple_spectrum(rect_mod(1:Nfft), fs);
[fx_srrc, S_srrc]   = simple_spectrum(srrc_mod_demo(1:Nfft), fs);

figure;
subplot(2,1,1);
plot(fx_rect/1e6, 20*log10(abs(S_rect)+1e-12), 'b'); grid on;
title('未成形（矩形脉冲）调制后的带通信号频谱');
xlabel('f / MHz'); ylabel('|X(f)| dB');

subplot(2,1,2);
plot(fx_srrc/1e6, 20*log10(abs(S_srrc)+1e-12), 'r'); grid on;
title('SRRC 脉冲成形后调制的带通信号频谱');
xlabel('f / MHz'); ylabel('|X(f)| dB');

%% ========= (C.real) 正式 DBPSK + SRRC 调制链路 =========
disp("正在生成正式 DBPSK + SRRC 调制信号...");

% 1) 全部 PCM bit → ±1
symbols_full = 2*bits_pcm - 1;

% 2) DBPSK 差分编码
diff_syms = zeros(size(symbols_full));
diff_syms(1) = symbols_full(1);
for k = 2:length(symbols_full)
    diff_syms(k) = diff_syms(k-1)*symbols_full(k);
end

% 3) 上采样
up_syms = upsample(diff_syms, Ns);

% 4) SRRC 脉冲成形（正式基带）
tx_bb = filter(srrc, 1, up_syms);    % 发射基带

% 5) 带通调制
t = (0:length(tx_bb)-1)/fs;
tx = tx_bb .* cos(2*pi*Fc*t);        % 发射带通

disp("正式 SRRC 调制信号生成完毕。");

% 正式已调信号频谱
[fx, SX] = simple_spectrum(tx, fs);


%% ========= (D) 加噪（AWGN 信道） =========
SNRdB = 20;  % 单点演示
tx_pow = mean(tx.^2);
npow   = tx_pow/10^(SNRdB/10);
noise  = sqrt(npow)*randn(size(tx));
rx   = tx + noise;
% rx = tx;    % 先看无噪声波形对齐

% 波形片段（时域对比）
seg = 1:min(4000, numel(t));
figure; plot(t(seg), tx(seg), 'b', t(seg), rx(seg), 'r'); grid on;
legend('tx','rx'); 
title(sprintf('DBPSK 已调与加噪后波形, SNR=%.1f dB',SNRdB));
xlabel('t/s');

%% ========= (D.X) 接收端：乘法器 × SRRC 匹配滤波 =========
t_rx = (0:length(rx)-1)/fs;
local_carrier = cos(2*pi*Fc*t_rx);

% 带通 → 基带
rx_bb = rx .* local_carrier;

% 匹配滤波（SRRC）
rx_matched = filter(srrc, 1, rx_bb);   % 两个 SRRC 级联

% 画 rx / rx×cos / 匹配后
figure;
Nplot2 = min(2000, length(rx));
subplot(3,1,1);
plot(rx(1:Nplot2),'b'); grid on;
title('接收信号 rx（带通）');

subplot(3,1,2);
plot(rx_bb(1:Nplot2),'m'); grid on;
title('rx × cos(2πF_ct)（乘法器输出，基带）');

subplot(3,1,3);
plot(rx_matched(1:Nplot2),'r'); grid on;
title('SRRC 匹配滤波输出（基带信号）');
xlabel('样本点');

%% ========= (E) DBPSK 解调（自写 SRRC 版） =========
% 使用匹配滤波后的 rx_matched 做差分解调 + 抽样判决

% 全长解调
bits_hat_all = dbpsk_srrc_demod(rx_matched, numel(bits_pcm), Ns, span);

% 对齐比特长度为 8 的整数倍
L_bits = floor(min(numel(bits_hat_all), numel(bits_pcm))/8)*8;
bits_hat = bits_hat_all(1:L_bits);
bits_ref = bits_pcm(1:L_bits);

%% ========= (E.X) SRRC 基带对齐 + 抽样点示意 =========
disp('绘制 SRRC 匹配滤波输出与发射基带、抽样点比较...');

tx_bb_full = tx_bb;
rx_bb_full = rx_matched;

% 发射基带到接收匹配输出只有一个 SRRC 的额外延迟 = span*Ns/2
delay_rx_only = span*Ns/2;

L_tx   = length(tx_bb_full);
L_rx   = length(rx_bb_full);
L_valid = min(L_tx, L_rx - delay_rx_only);

if L_valid > 0
    tx_aligned = tx_bb_full(1:L_valid);
    rx_aligned = rx_bb_full(delay_rx_only+1 : delay_rx_only+L_valid);

    Nplot3 = min(3000, L_valid);
    figure;
    plot(tx_aligned(1:Nplot3),'b'); hold on;
    plot(rx_aligned(1:Nplot3),'r'); grid on;
    legend('TX：SRRC 脉冲成形基带','RX：SRRC 匹配滤波输出（对齐后）');
    title('SRRC 发射基带 vs 接收匹配滤波基带 对比');
    xlabel('样本点');
end

% ==== 抽样位置（用于示意） ====
start_pos = delay_rrc + 1;                  % 第一个符号采样点
sample_pos = start_pos + (0:length(bits_ref)-1)*Ns;
sample_pos = sample_pos(sample_pos <= length(rx_bb_full));

rx_samples = rx_bb_full(sample_pos);
rx_decision = rx_samples > 0;


%% ========= (F) BER 计算 =========
BER_now = mean(xor(bits_ref, bits_hat));
fprintf('单点 SNR = %.1f dB 时，BER = %.4e\n', SNRdB, BER_now);

%% ========= (FX) 四图对比：PCM 波形/比特 vs 解码 =========
disp("绘制『四图对比』：PCM波形、SRRC波形、PCM比特、解调比特...");


Nplot_pcm = min(500, length(sampleData));

% ① SRRC 匹配滤波后的基带波形 + 抽样点
figure;
Nplot_srrc = min(5000, length(rx_matched));
subplot(1,1,1);
plot(rx_matched(1:Nplot_srrc),'m'); hold on; grid on;
title(' SRRC 匹配滤波基带信号 + 抽样点标注');
xlabel('样本编号'); ylabel('幅度');

sample_pos_valid = sample_pos(sample_pos <= Nplot_srrc);
stem(sample_pos_valid, rx_matched(sample_pos_valid),'r','filled');
legend('SRRC 基带','抽样点');

% ② PCM 原始比特
Nshow_bits = min(80, length(bits_ref));
figure;
subplot(2,1,1);
stem(1:Nshow_bits, bits_ref(1:Nshow_bits),'b','LineWidth',1.2);
grid on;
title(' PCM 原始比特（前 80 位）');
xlabel('符号编号'); ylabel('bit');
xlim([1 Nshow_bits]);

% ③ DBPSK 解调最终比特
subplot(2,1,2);
stem(1:Nshow_bits, bits_hat(1:Nshow_bits),'r','LineWidth',1.2);
grid on;
title(' DBPSK+SRRC 解调比特（前 80 位）');
xlabel('符号编号'); ylabel('bit');
xlim([1 Nshow_bits]);

%% ========= (G) 眼图（直接用 rx_matched） =========
Leye = floor((length(rx_matched) - delay_rrc)/Ns) * Ns;
if Leye > 0
    % 截取有效数据（去除滤波器延迟部分）
    valid_eye_data = rx_matched(delay_rrc+1:delay_rrc+Leye);
    
    % 关键修改：对眼图数据进行循环移位半个符号
    shift_samples = round(Ns/2);  % 半个符号的样本数
    shifted_eye_data = circshift(valid_eye_data, -shift_samples);
    
    % 重新整形为眼图矩阵
    eye_mat = reshape(shifted_eye_data, Ns, []);
    
    figure;
    plot(eye_mat(:, 1:min(40, size(eye_mat,2))), 'b-', 'LineWidth', 0.5); 
    grid on;
    title('接收端眼图（时间对齐后）');
    xlabel('每个符号内的样本点'); 
    ylabel('幅度');
    
    % 标记最佳采样点
    hold on;
    plot([Ns/2, Ns/2], [min(shifted_eye_data), max(shifted_eye_data)], 'r--', 'LineWidth', 2);
    legend('眼图轨迹', '最佳采样点', 'Location', 'best');
    hold off;
end
%% ========= (H) 频谱：输入/输出模拟信号 =========


% PCM 解码回"模拟"信号（8kHz 域）
outData = PCM_13Decode(bits_hat);
t_out = (0:numel(outData)-1)/sampleVal;

figure; plot(t_smp, sampleData,'b'); hold on;
plot(t_out, outData,'r'); grid on;
legend('PCM 抽样后(发送侧)','PCM 解码后(接收侧)');
title('输入/输出 "模拟" 波形（8kHz 域）'); xlabel('t/s');

figure;
subplot(2,1,1);
[fa, SA] = simple_spectrum(y_analog, Fs_in);
plot(fa/1e3, 20*log10(abs(SA)+1e-12)); grid on;
xlabel('f / kHz'); ylabel('|S(f)| dB'); 
title('输入模拟信号频谱');
subplot(2,1,2);
[fo, SO] = simple_spectrum(outData, sampleVal);
plot(fo/1e3, 20*log10(abs(SO)+1e-12)); grid on;
xlabel('f / kHz'); ylabel('|S(f)| dB'); 
title('输出模拟信号频谱（解码后）');
%% ========= (I) BER 曲线：SNR 扫描 =========
SNRs = 0:2:14;
BERs = zeros(size(SNRs));

for kk = 1:numel(SNRs)
    EbN0_dB = SNRs(kk);            % 现在把横坐标理解为 Eb/N0
    EbN0    = 10^(EbN0_dB/10);

    % 关键修改：为了得到指定的 Eb/N0，噪声功率要多乘一个 Ns
    npow_k  = tx_pow * Ns / EbN0;  % 和你原来相比，多了一个 Ns

    n_k = sqrt(npow_k)*randn(size(tx));
    r_k = tx + n_k;

    t_rx_k = (0:length(r_k)-1)/fs;
    local_carrier_k = cos(2*pi*Fc*t_rx_k);
    r_bb = r_k .* local_carrier_k;
    r_matched = filter(srrc, 1, r_bb);

    bhat_all = dbpsk_srrc_demod(r_matched, numel(bits_pcm), Ns, span);
    Lk = floor(min(numel(bhat_all), numel(bits_pcm))/8)*8;
    BERs(kk) = mean(xor(bits_pcm(1:Lk), bhat_all(1:Lk)));
end

figure; semilogy(SNRs, BERs, '-o'); grid on;
xlabel('E_b/N_0 / dB'); ylabel('BER');
title('DBPSK + SRRC 误码率曲线');
ylim([1e-5 1]);


%% ========= 本脚本结束，下面是局部函数 =========
function bits_hat = dbpsk_srrc_demod(rx_matched, bit_num, Ns, span)
% 基于 SRRC 匹配滤波输出的简单 DBPSK 解调
% rx_matched : 匹配滤波输出
% bit_num    : 需要解调的比特数（通常 = length(bits_pcm)）
% Ns         : 每个符号采样点
% span       : SRRC span（符号数）

    delay_rrc = span * Ns;          % 两个 SRRC 级联的总群时延
    start_idx = delay_rrc + 1;      % 第一个有效符号采样点

    sample_pos = start_idx:Ns:length(rx_matched);
    sample_pos = sample_pos(1:min(bit_num, numel(sample_pos)));

    rx_smpl = rx_matched(sample_pos);
    d_hat = sign(rx_smpl);
    d_hat(d_hat == 0) = 1;          % 避免 0

    N = numel(d_hat);
    sym_hat  = zeros(1, N);
    bits_hat = zeros(1, N);

    % 第一个符号直接判决
    sym_hat(1)  = d_hat(1);
    bits_hat(1) = (sym_hat(1)+1)/2;

    % 后续符号：d(k)*d(k-1) = 原来的 (2b-1)
    for ii = 2:N
        sym_hat(ii)  = d_hat(ii) * d_hat(ii-1);
        bits_hat(ii) = (sym_hat(ii)+1)/2;
    end

    bits_hat = bits_hat(:).';       % 保证行向量
end
