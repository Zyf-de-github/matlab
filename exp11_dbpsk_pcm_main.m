clc; clear; close all;
addpath('./pcm');
%% ========= 第13组参数 =========
Rb = 64e3;                 % 比特率 
Fc = 10*Rb;                % 载波
fs = 12*Fc;                % 采样率（注意：较高，内存别设太长序列）
Ns = fs/Rb;                % 每比特采样点（应为整数：这里=120）
assert(abs(Ns-round(Ns))<1e-12,'fs/Rb 必须为整数'); Ns = round(Ns);

%% ========= (A) 生成模拟输入：3正弦叠加 =========
Fs_in = 48e3;                          % 生成"模拟信号"的采样率
Tsig  = 0.25;                          % 信号时长，秒（可调）
t_in  = (0:1/Fs_in:Tsig-1/Fs_in);

% --- 分别的三路 ---
f1=3; f2=10; f3=20;                     % 三个频率
a1=0.8; a2=0.5; a3=0.3;                 % 不同幅度

sig1 = a1*sin(2*pi*f1*t_in);
sig2 = a2*sin(2*pi*f2*t_in);
sig3 = a3*sin(2*pi*f3*t_in);

% --- 合成 ---
y_analog = sig1 + sig2 + sig3;
y_analog = y_analog/max(abs(y_analog))*0.98;   % 归一防溢

% --- 画图 ---
figure;
plot(t_in, sig1, 'r'); hold on;
plot(t_in, sig2, 'g');
plot(t_in, sig3, 'b');
plot(t_in, y_analog, 'W','LineWidth',1.2);
grid on;

legend('sin1','sin2','sin3','sum');
title('输入模拟信号（3正弦叠加 + 单独显示三路）');
xlabel('t/s'); ylabel('amp');
%% ========= (B) PCM 13折线编码（8kHz） =========
sampleVal = Rb/8; % 8 kHz ⇒ 8 bit/样本 ⇒ 64 kbps，刚好匹配 Rb
[sampleData, bits_pcm] = PCM_13Encode(y_analog, Fs_in, sampleVal);

% 画PCM抽样后的波形与bit流
t_smp = (0:numel(sampleData)-1)/sampleVal;
figure; plot(t_smp, sampleData); grid on;
title('PCM抽样后波形'); xlabel('t/s');
figure; stairs(bits_pcm(1:100)); ylim([-0.2 1.2]); grid on;
title('PCM编码后bit（前400位）');

%% ========= (C) DBPSK 调制 =========
[t, tx] = DBPSK_Modulation(bits_pcm, Fc, fs, Rb);


%% ========= (C.X) 码型变换：矩形脉冲 vs SRRC 脉冲 =========

% 取前 20 个比特做脉冲展示（太多画不清楚）
Nshow = 20;
bps = bits_pcm(1:Nshow);

% 映射为 ±1
symbols = 2*bps - 1;

% ========== 1) 矩形脉冲（未成形）==========
rect = upsample(symbols, Ns);  % 上采样后保持一个矩形符号
rect_wave = filter(ones(1, Ns), 1, rect); % 矩形成形（累加）

% ========== 2) SRRC 脉冲（根升余弦）==========
beta = 0.75;
span = 8;
srrc = rcosdesign(beta, span, Ns, 'sqrt');
srrc_wave = filter(srrc, 1, upsample(symbols, Ns));

% ========== 画两者对比 ==========
figure;
subplot(2,1,1);
plot(rect_wave, 'b'); grid on;
title('矩形脉冲成形（矩形滤波器）');
ylabel('幅度');
xlim([0, Nshow*Ns]);

subplot(2,1,2);
plot(srrc_wave, 'r'); grid on;
title('SRRC 脉冲成形（根升余弦）');
xlabel('样本点'); ylabel('幅度');
xlim([0, Nshow*Ns]);
%% ========= (C.Y) 乘法器：脉冲成形后的调制（基带 → 带通） ==========

% 时间轴匹配（取与 rect_wave 相同长度）
t_show = (0:length(rect_wave)-1)/fs;

% 载波
carrier = cos(2*pi*Fc*t_show);

% 1) 矩形脉冲 × 载波
rect_mod = rect_wave .* carrier;

% 2) SRRC 脉冲 × 载波
srrc_mod = srrc_wave .* carrier;

% 仅画前 Nplot 个样本（为了清晰对比）
Nplot = min(2000, length(rect_wave));   % 自动限制2000点

%% ======== 绘图：调制前 vs 调制后 ========
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
plot(srrc_wave(1:Nplot),'r'); grid on;
title('SRRC 脉冲（基带）');
xlabel('样本点'); ylabel('幅度');

subplot(2,2,4);
plot(srrc_mod(1:Nplot),'r'); grid on;
title('SRRC 脉冲 × cos(2πF_ct)（带通）');
xlabel('样本点');


%============================================================================
% 分离 I/Q 信号（实部是 I 路，虚部是 Q 路）
%CH1 = real(tx);  % I 路信号
%CH2 = imag(tx);  % Q 路信号

% 设置 DA 输出的参数
%divFreq = 30720000 / fs - 1;  % 分频值，确保 fs 与分频值匹配
%dataNum = length(tx);         % 数据长度，应该与 tx 信号的长度匹配
%isGain = 1;                   % 是否增益，设置为 1 表示增益

% 调用 DA 输出函数，发送 I/Q 信号到硬件
%DA_OUT(CH1, CH2, divFreq, dataNum, isGain);
%============================================================================
% 频谱（已调）
[fx, SX] = simple_spectrum(tx, fs);
figure; plot(fx/1e6, 20*log10(abs(SX)+1e-12)); grid on;
xlabel('f / MHz'); ylabel('|X(f)| dB'); title('已调信号幅度谱（dB）');

%% ========= (C.Z) SRRC 调制 vs 未成形调制：频谱对比 =========

% 自动选择 FFT 点数（不超过信号长度）
Nfft = min([4096, length(rect_mod), length(srrc_mod)]);

% 1) 未成形矩形脉冲 × 载波
[fx_rect, S_rect] = simple_spectrum(rect_mod(1:Nfft), fs);

% 2) SRRC 脉冲成形 × 载波
[fx_srrc, S_srrc] = simple_spectrum(srrc_mod(1:Nfft), fs);

figure;
subplot(2,1,1);
plot(fx_rect/1e6, 20*log10(abs(S_rect)+1e-12), 'b'); grid on;
title('未成形（矩形脉冲）调制后的带通信号频谱');
xlabel('f / MHz'); ylabel('|X(f)| dB');

subplot(2,1,2);
plot(fx_srrc/1e6, 20*log10(abs(S_srrc)+1e-12), 'r'); grid on;
title('SRRC 脉冲成形后调制的带通信号频谱');
xlabel('f / MHz'); ylabel('|X(f)| dB');




%% ========= (D) 加噪（自写AWGN，不用工具箱） =========
SNRdB = 10;  % 单次演示用
tx_pow = mean(tx.^2);
npow   = tx_pow/10^(SNRdB/10);
noise  = 0.2*sqrt(npow)*randn(size(tx));
% rx     = tx + noise;
rx=tx;

% 波形片段
seg = 1:min(4000, numel(t));
figure; plot(t(seg), tx(seg), 'b', t(seg), rx(seg), 'r'); grid on;
legend('tx','rx'); title(sprintf('DBPSK已调与加噪后波形, SNR=%.1f dB',SNRdB));
xlabel('t/s');

%% ========= (D.X) 接收端：乘法器 × SRRC 匹配滤波 =========

% (1) 本地载波
t_rx = (0:length(rx)-1)/fs;
local_carrier = cos(2*pi*Fc*t_rx);

% (2) 乘法器：带通 → 基带
rx_bb = rx .* local_carrier;

% (3) SRRC 匹配滤波（注意用与发送端相同的 srrc）
rx_matched = filter(srrc, 1, rx_bb);

% (4) 画对比：带通 vs 乘法器后 vs SRRC 匹配滤波后
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


%% ========= (E) DBPSK 解调（差分乘积+滑动平均） =========
[bits_hat, eye_buf] = DBPSK_Demod(rx_matched, fs, Rb);
%============================================================================
% 分离 I/Q 信号（解调后的信号）
%CH1 = real(rx);  % I 路信号
%CH2 = imag(rx);  % Q 路信号

% 设置 DA 输出的参数
%divFreq = 30720000 / fs - 1;  % 分频值，确保与系统设置匹配
%dataNum = length(rx);         % 数据长度，应该与接收信号的长度一致
%isGain = 1;                   % 是否增益，设置为 1 表示增益

% 调用 DA 输出函数，发送解调后的信号到硬件
%DA_OUT(CH1, CH2, divFreq, dataNum, isGain);  % 发送 I/Q 信号到硬件
%============================================================================
% 计算并显示解调后的误码率（BER）
L = floor(numel(bits_hat) / 8) * 8;    % 对齐到8的整数倍
bits_hat = bits_hat(1:L);               % 截取到8的整数倍长度
bits_ref = bits_pcm(1:L);               % 参考比特流对齐

%% ========= (F) BER 计算 =========
BER_now = mean(xor(bits_ref, bits_hat));
fprintf('单点SNR=%.1f dB时，BER=%.4e\n', SNRdB, BER_now);

%% ========= (G) SRRC 匹配滤波后的眼图 =========
Ns = fs/Rb;
eye_sig = eye_buf(1:Ns*40);   % 截取40个符号
eye_mat = reshape(eye_sig, Ns, []);
figure;
plot(eye_mat, 'y'); grid on;
title('SRRC 匹配滤波后的眼图');
xlabel('样本点'); ylabel('幅度');


%% ========= (H) 频谱：关键数字信号 与 输入/输出模拟信号 =========
% 输入模拟信号频谱（原48k）
[fa, SA] = simple_spectrum(y_analog, Fs_in);
figure; plot(fa/1e3, 20*log10(abs(SA)+1e-12)); grid on;
xlabel('f / kHz'); ylabel('|S(f)| dB'); title('输入模拟信号频谱');

% 解码回的模拟信号
outData = PCM_13Decode(bits_hat);
t_out = (0:numel(outData)-1)/sampleVal;

figure; plot(t_smp, sampleData,'b'); hold on;
plot(t_out, outData,'r'); grid on;
legend('PCM抽样后(发送侧)','PCM解码后(接收侧)');
title('输入/输出"模拟"波形（8kHz域）'); xlabel('t/s');

[fo, SO] = simple_spectrum(outData, sampleVal);
figure; plot(fo/1e3, 20*log10(abs(SO)+1e-12)); grid on;
xlabel('f / kHz'); ylabel('|S(f)| dB'); title('输出模拟信号频谱（解码后）');

%% ========= (I) BER曲线：SNR扫描 =========
SNRs = 0:2:14;  BERs = zeros(size(SNRs));
for k=1:numel(SNRs)
    npow = tx_pow/10^(SNRs(k)/10);
    n = sqrt(npow)*randn(size(tx));
    r = tx + n;
    [bhat,~] = DBPSK_Demod(r, fs, Rb);
    L = floor(min(numel(bhat), numel(bits_pcm))/8)*8;
    BERs(k) = mean(xor(bits_pcm(1:L), bhat(1:L)));
end
figure; semilogy(SNRs, BERs,'-o'); grid on;
xlabel('SNR / dB'); ylabel('BER'); title('DBPSK 误码率曲线');

%% ========= (J) （可选）硬件DA输出 =========
% 若要上板到示波器看I/Q：
% CH1 = rx(1:2:end); CH2 = rx(2:2:end);  % 例：随便拆出两路
% divFreq = 30720000/fs - 1;  % 参考你们板卡约束
% isGain = 1;
% DA_OUT(CH1, CH2, divFreq, length(CH1), isGain);
