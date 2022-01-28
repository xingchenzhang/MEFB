% The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricWang.m
% and https://github.com/zhengliu6699/imageFusionMetrics/blob/master/NCC.m
% The interface is modified by the author of MEFB
%
% References for metrics
% Q. Wang, Y. Shen, J.Q. Zhang, A nonlinear correlation measure for multivariable data set, Physica D 200 (3¨C4) (2005) 287¨C295.
% Q. Wang, Y. Shen, J. Jin, Performance evaluation of image fusion techniques, Image Fusion Algorithms Appl. 19 (2008) 469¨C492.

function res = metricsQncie(img1,img2,fused) 

    [m,n,b] = size(fused); 
    [m1,n1,b1] = size(img1);
    [m2,n2,b2] = size(img2);
    
    if (b1 == 1) && (b2 ==1) && (b == 3)
        fused_new = zeros(m,n);
        fused_new = fused(:,:,1);
        fused = fused_new;
    end
    [m,n,b] = size(fused); 

    if b == 1
        g = Qncie(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
            g(k) = Qncie(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Qncie(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end

function output=Qncie(im1,im2,fused)

    % function res=metricWang(im1,im2,fused)
    %
    % This function implements Wang's algorithms for fusion metric.
    % im1, im2 -- input images;
    % fused      -- fused image;
    % res      -- metric value;
    %
    % IMPORTANT: The size of the images need to be 2X. 
    % See also: NCC.m, mutual_info.m, evalu_fusion.m
    %
    % Z. Liu [July 2009]
    %

    % Ref: Performance evaluation of image fusion techniques, Chapter 19, pp.469-492, 
    % in Image Fusion:  Algorithms and Applications, edited by Tania Stathaki
    % by Qiang Wang
    %

    %% pre-processing
    im1=normalize1(im1);
    im2=normalize1(im2);
    fused=normalize1(fused);

    [hang,lie]=size(im1);
    b=256;
    K=3;

    %% Call mutual_info.m
    % two inputs

    NCCxy=NCC(im1,im2);


    % one input and fused image
    NCCxf=NCC(im1,fused);


    % another input and fused image
    NCCyf=NCC(im2,fused);


    %% get the correlation matrix and eigenvalue 

    R=[ 1 NCCxy NCCxf; NCCxy 1 NCCyf; NCCxf NCCyf 1];
    r=eig(R);

    %% HR

    HR=sum(r.*log2(r./K)/K);
    HR=-HR/log2(b);

    %% NCIE

    NCIE=1-HR;

    output=NCIE;

end

function res=NCC(im1,im2)

% function res=NCC(im1,im2)
% 
% This function is caculate the mutual information of two input images.
% im1   -- input image one;
% im2   -- input image two;
%
% res    -- NNC (nonlinear correlation coefficient
%
%
% Note: 1) The input images need to be in the range of 0-255. (see function:
% normalize1.m); 2) This function is similar to mutual information but they
% are different.
%
% Z. Liu @ NRCC [July 17, 2009]

im1=double(im1);
im2=double(im2);

[hang,lie]=size(im1);
count=hang*lie;
N=256;
b=256;

%% caculate the joint histogram
h=zeros(N,N);

for i=1:hang
    for j=1:lie
        % in this case im1->x (row), im2->y (column)
        h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
    end
end

%% marginal histogram

% this operation converts histogram to probability
%h=h./count;
h=h./sum(h(:));

im1_marg=sum(h);    % sum each column for im1
im2_marg=sum(h');   % sum each row for im2


%for i=1:N
%    if (im1_marg(i)>eps)
%        % entropy for image1
%        Hx=Hx+(-im1_marg(i)*(log2(im1_marg(i))));
%    end
%    if (im2_marg(i)>eps)
%        % entropy for image2
%        Hy=Hy+(-im2_marg(i)*(log2(im2_marg(i))));
%    end
%end

H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));


% joint entropy

%H_xy=0;

%for i=1:N
%    for j=1:N
%        if (h(i,j)>eps)
%            H_xy=H_xy+h(i,j)*log2(h(i,j));
%        end
%    end
%end

H_xy=-sum(sum(h.*log2(h+(h==0))));
H_xy=H_xy/log2(b);


%H_xy=-sum(sum(h.*(log2(h+(h==0)))));

H_x=H_x/log2(b);
H_y=H_y/log2(b);

% NCC
res=H_x+H_y-H_xy;

end