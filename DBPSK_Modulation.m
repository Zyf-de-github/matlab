function [t, tx_signal] = DBPSK_Modulation(bits, Fc, fs, Rb)
    Tb = 1/Rb;  Ns = fs/Rb;
    assert(abs(Ns-round(Ns))<1e-12,'fs/Rb 必须为整数'); Ns = round(Ns);
    N  = numel(bits);
    t  = 0:1/fs:(N*Tb - 1/fs);

    % 差分相位：1->翻转π, 0->不变；累加得到每符号绝对相位
    dphi      = pi*(bits(:).'>0);
    phase_sym = mod(cumsum(dphi), 2*pi);       % 约束到 [0,2π)
    phase_smp = repelem(phase_sym, Ns);        % 扩展到采样级

    tx_signal = cos(2*pi*Fc*t + phase_smp);    % 实带通DBPSK
end
