function [bits_hat, eye_buf] = DBPSK_Demod(rx, Fs, Rb)
    Ns = Fs/Rb; Ns = round(Ns);

    % 符号间差分乘积（不显式载波同步）
    y = [zeros(1,Ns), rx(1:end-Ns)] .* rx;     % r[k]*r[k-Ns]

    % 简单低通/匹配：滑动平均（长度≈Ns/2~Ns）
    L = max(8, round(Ns/2));  h = ones(1,L)/L;
    z = conv(y, h, 'same');                    % 平滑后的差分能量

    % 抽样判决：每符号中点
    sp = round((Ns/2):Ns:(numel(z)-Ns/2));
    z_s = z(sp);
    bits_hat = z_s < 0;                        % <0 判为1（发生π翻转）

    % 输出眼图缓冲（把z当作基带能量轨迹来画眼图）
    eye_buf = z;
end
