%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  FileName:            a_13coding.m
%  Description:         PCM 13折线语音编码
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  Parameter List:       
%       Output Parameter
%           a13_moddata 编码后的bit流数据
%       Input Parameter
%           x           输入语音信号抽样后的数据
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function  [ a13_moddata ] = a_13coding( x )
n=length(x);
a13_moddata=zeros(1,n*8);
for bb=1:n
    Is=x(1,bb);
    if Is>1||Is<-1,error('input must within [-1,1]'),end
    Is=round(Is*2048);
    C=zeros(1,8);  %将8位PCM编码初始化为全0
    if Is>0
        C(1)=1 ;  %判断抽样值的正负
    end
    
    % the polarity determins C(1)
    abIs=abs(Is);
    
    if 0<abIs && abIs<=16
        C(2:4)=[0 0 0];    %8级量化编码
        q=1;
        a=0;
        C(5:8)=e_coding(abIs,q,a); %16级量化编码
    end
     if 16<abIs && abIs<=32
        C(2:4)=[0 0 1];
        q=1;
        a=16;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 32<abIs && abIs<=64
        C(2:4)=[0 1 0];
        q=2;
        a=32;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 64<abIs && abIs<=128
        C(2:4)=[0 1 1];
        q=4;
        a=64;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 128<abIs && abIs<=256
        C(2:4)=[1 0 0];
        q=8;
        a=128;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 256<abIs && abIs<=512
        C(2:4)=[1 0 1];
        q=16;
        a=256;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 512<abIs && abIs<=1024
        C(2:4)=[1 1 0];
        q=32;
        a=512;
        C(5:8)=e_coding(abIs,q,a);
    end
    if 1024<abIs && abIs<=2048
        C(2:4)=[1 1 1];
        q=64;
        a=1024;
        C(5:8)=e_coding(abIs,q,a);
    end
   
      a13_moddata(1,(bb-1)*8+1:bb*8)=C;  %得到8位pcm编码
end
end



