%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName:            main.m
%  Description:         PCM 13折线编解码
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%    需编制代码 建立输入模拟信号ych1





figure
dt=1/Fs;
t=0:dt:(length(yCh1)-1)*dt;
plot(t,yCh1);
title('输入信号波形');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%PCM 13折线编码
sampleVal=8000;%8k抽样率
[sampleData,a13_moddata]=PCM_13Encode(yCh1,Fs,sampleVal);

figure
dt1=1/sampleVal;
t1=0:dt1:(length(sampleData)-1)*dt1;
plot(t1,sampleData);
title('输入信号抽样后的波形');

figure
plot(a13_moddata);
title('编码后的bit数据');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%PCM 13折线解码 
[outData] = PCM_13Decode( a13_moddata );

figure
dt1=1/sampleVal;
t1=0:dt1:(length(sampleData)-1)*dt1;
plot(t1,outData);
title('解码还原后的波形');


