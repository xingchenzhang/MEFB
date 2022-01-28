% The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricYang.m
% The interface is modified by the authors of MEFB
% 
% Reference for this metric
% C. Yang, J.-Q. Zhang, X.-R. Wang, X. Liu, A novel similarity based quality metric for 
% image fusion, Inf. Fusion 9 (2) (2008) 156¨C160.

function res = metricsQy(img1,img2,fused) 

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
        g = Qy(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
            g(k) = Qy(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Qy(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end


function output = Qy(im1,im2,fused)

% function res=metricYang(im1,im2,fused)
%
% This function implements Yang's algorithm for fusion metric.
% im1, im2 -- input images;
% fused      -- fused image;
% res      -- metric value.
%
% Z. Liu [July 2009]
%

% Ref: A novel similarity based quality metric for image fusion, Information Fusion, Vol.9, pp156-160, 2008
% by Cui Yang et al.
% 

%% some pre-processing

    s=size(size(im1));
    if s(2)==3 
        im1=rgb2gray(im1);
    else
        im1=im1;
    end 

    s1=size(size(im2));
    if s1(2)==3 
        im2=rgb2gray(im2);
    else
        im2=im2;
    end 
    
    s2=size(size(fused));
    if s2(2)==3 
        fused=rgb2gray(fused);
    else
        fused=fused;
    end 
    
    im1=double(im1);
    im2=double(im2);
    fused=double(fused);


%% first call the SSIM 
[mssim1, ssim_map1, sigma1_sq1,sigma2_sq1] = ssim_yang(im1, im2);
[mssim2, ssim_map2, sigma1_sq2,sigma2_sq2] = ssim_yang(im1, fused);
[mssim3, ssim_map3, sigma1_sq3,sigma2_sq3] = ssim_yang(im2, fused);

bin_map=ssim_map1>=0.75;
ramda=sigma1_sq1./(sigma1_sq1+sigma2_sq1);

Q1=(ramda.*ssim_map2+(1-ramda).*ssim_map3).*bin_map;

Q2=(max(ssim_map2,ssim_map3)).*(~bin_map);

A = Q1+Q2;
A=A(~isnan(A));
Q = mean2(A);

output=Q;
end


%%
function [mssim, ssim_map, sigma1_sq,sigma2_sq] = ssim_yang(img1, img2)

%========================================================================

[M N] = size(img1);
if ((M < 11) | (N < 11))
   ssim_index = -Inf;
   ssim_map = -Inf;
  return
end
window = fspecial('gaussian', 7, 1.5);	%

L = 255;                                  %

C1 = 2e-16;
C2 = 2e-16;

window = window/sum(sum(window));
img1 = double(img1);
img2 = double(img2);
mu1   = filter2(window, img1, 'valid');
mu2   = filter2(window, img2, 'valid');
mu1_sq = mu1.*mu1;
mu2_sq = mu2.*mu2;
mu1_mu2 = mu1.*mu2;
sigma1_sq = filter2(window, img1.*img1, 'valid') - mu1_sq;
sigma2_sq = filter2(window, img2.*img2, 'valid') - mu2_sq;
sigma12 = filter2(window, img1.*img2, 'valid') - mu1_mu2;
if (C1 > 0 & C2 > 0)
   ssim_map = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))./((mu1_sq + mu2_sq + C1).*(sigma1_sq + sigma2_sq + C2));
else
   numerator1 = 2*mu1_mu2 + C1;
   numerator2 = 2*sigma12 + C2;
    denominator1 = mu1_sq + mu2_sq + C1;
   denominator2 = sigma1_sq + sigma2_sq + C2;
   ssim_map = ones(size(mu1));
   
   index = (denominator1.*denominator2 > 0);
   ssim_map(index) = (numerator1(index).*numerator2(index))./(denominator1(index).*denominator2(index));
   index = (denominator1 ~= 0) & (denominator2 == 0);
   ssim_map(index) = numerator1(index)./denominator1(index);
end

mssim = mean2(ssim_map);


%return
end
