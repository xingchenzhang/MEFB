% The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricCvejic.m
% The interface is modified by the author of MEFB
%
% Reference for the metric
% N. Cvejic, A. Loza, D. Bull, N. Canagarajah, A similarity metric for assessment of image fusion 
% algorithms, Int. J. Signal Process. 2 (3) (2005) 178¨C182.

function res = metricsQc(img1,img2,fused) 

    % Get the size of img 
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
        g = Qc(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
            g(k) = Qc(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Qc(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end


function output = Qc(im1,im2,fused)

% function res=metricCvejic(im1,im2,fused,sw)
%
% This function implements Chen's algorithm for fusion metric.
% im1, im2 -- input images;
% fused      -- fused image;
% res      -- metric value;
% sw       -- 1: metric 1; 2: metric 2. Cvejic has two different metics.
%
% IMPORTANT: The size of the images need to be 2X. 
%
% Z. Liu [July 2009]
%

% Ref: Metric for multimodal image sensor fusion, Electronics Letters, 43 (2) 2007 
% by N. Cvejic et al.
%
% Ref: A Similarity Metric for Assessment of Image Fusion Algorithms, International Journal of Information and Communication Engineering 2 (3) 2006, pp.178-182.
% by N. Cvejic et al.
%


%% pre-processing
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


    [mssim2, ssim_map2, sigma_XF] = ssim_yang(im1, fused);
    [mssim3, ssim_map3, sigma_YF] = ssim_yang(im2, fused);

    simXYF=sigma_XF./(sigma_XF+sigma_YF);
    sim=simXYF.*ssim_map2+(1-simXYF).*ssim_map3;


    index=find(simXYF<0);
    sim(index)=0;

    index=find(simXYF>1);
    sim(index)=1;

    sim=sim(~isnan(sim));        

    output=mean2(sim);

end

%%
function [mssim, ssim_map, sigma12] = ssim_yang(img1, img2)

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
