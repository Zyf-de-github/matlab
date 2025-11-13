function [f, X] = simple_spectrum(x, fs)
    N = numel(x);
    win = hann(N).';                             % 抗泄漏（可换成 ones）
    xw = x(:)'.*win;
    Xf = fftshift(fft(xw))/N;
    f  = (-N/2:N/2-1)*(fs/N);
    X  = Xf;
end
