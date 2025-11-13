%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName      : DA_OUT_2025b.m
%  Description   : 适配 MATLAB 2025b 的 DA 输出函数
%                  已完成：udp → udpport；fwrite → write
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = DA_OUT(CH1_data,CH2_data,divFreq,dataNum,isGain)

ROM_MAX_LEN=30720;
ROM_MIN_LEN=100;
if dataNum>ROM_MAX_LEN
    disp('数据长度，超过长度限制范围100～30720');
    dataNum=ROM_MAX_LEN;
end
if  dataNum<ROM_MIN_LEN
    disp('数据长度，小于限制范围100～30720');
    dataNum=ROM_MIN_LEN;
end

if (divFreq>1024)||(divFreq<0)
    disp('fs采样率参数配置超过允许范围');
    divFreq=0;
end

temp_data=zeros(1,30720);
CH1_data_temp= [CH1_data ,temp_data];
CH2_data_temp= [CH2_data ,temp_data];
CH1_out_data=CH1_data_temp(1,1:dataNum);
CH2_out_data=CH2_data_temp(1,1:dataNum);

test_Set_router = uint8(hex2dec({'00','00','99','bb', '68','00','00','06',  '00','00','00','00',  '00','00','00','00',  '00','00','00','00'}));

divFreqL=mod(divFreq,256);
divFreqH=(divFreq-divFreqL)/256;
divFreqL=dec2hex(divFreqL);
divFreqH=dec2hex(divFreqH);

dataNumL=mod(dataNum,256);
dataNumH=(dataNum-dataNumL)/256;
dataNumL=dec2hex(dataNumL);
dataNumH=dec2hex(dataNumH);

test_tx_command = uint8(hex2dec({'00','00','99','bb', '65','0A','03','ff',  divFreqH,divFreqL,dataNumH,dataNumL,  '00','00','00','00',  '00','00','00','00'}));
test_Send_IQ    = uint8(hex2dec({'00','00','99','bb', '64','00','00','00',  '00','00','00','00',  '00','00','00','00',  '00','00','00','00'}));

SAMPLE_LENGTH = dataNum;
SEND_PACKET_LENGTH = 180;

% ---------- 2025b 创建udpport ----------
u = udpport("datagram","IPV4","LocalHost","192.168.1.21","LocalPort",12345,"Timeout",100);

dataIQ = zeros(1,SAMPLE_LENGTH*2);
dataIQ(1,1:2:end) = CH1_out_data(1,:);
dataIQ(1,2:2:end) = CH2_out_data(1,:);
if isGain==1
    dataIQ = dataIQ.*(2047/max(abs(dataIQ)));
end
dataIQ = fix(dataIQ);

for n = 1 : SAMPLE_LENGTH*2
    if dataIQ(n) > 2047
        dataIQ(n) = 2047;
    elseif  dataIQ(n) < 0
        dataIQ(n) = 4096 + dataIQ(n);
    end
end

dataIQ(1,1:2:SAMPLE_LENGTH*2-1) = dataIQ(1,1:2:SAMPLE_LENGTH*2).*16;
dataIQ(1,2:2:SAMPLE_LENGTH*2)   = fix(dataIQ(1,2:2:SAMPLE_LENGTH*2)./256) + rem(dataIQ(1,2:2:SAMPLE_LENGTH*2),256).*256;
dataIQ = uint16(dataIQ);

% ---------- 2025b write() API ----------
write(u,test_Set_router,"uint8","192.168.1.121",13345);
write(u,test_tx_command,"uint8","192.168.1.121",13345);
write(u,test_Send_IQ,"uint8","192.168.1.121",13345);

if SAMPLE_LENGTH*2<SEND_PACKET_LENGTH
    write(u,dataIQ(1,1:(SAMPLE_LENGTH*2)),"uint16","192.168.1.121",13345);
else
    for pn = 1:fix(SAMPLE_LENGTH*2/SEND_PACKET_LENGTH)
        write(u,dataIQ(1,((pn-1)*SEND_PACKET_LENGTH+1) : (pn*SEND_PACKET_LENGTH)),"uint16","192.168.1.121",13345);
    end
    write(u,dataIQ(1,(pn*SEND_PACKET_LENGTH+1):(SAMPLE_LENGTH*2)),"uint16","192.168.1.121",13345);
end

clear u; % 2025b 关闭

end
