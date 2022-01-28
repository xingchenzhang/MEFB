% The code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricPeilla.m
% and https://github.com/zhengliu6699/imageFusionMetrics/blob/master/ssim_index.m
% The interface is modified by the author of MEFB
%
% Reference for this metric
% G. Piella, H. Heijmans, A new quality metric for image fusion, in: Proceedings of International 
% Conference on Image Processing, Vol. 3, IEEE, 2003, pp. III¨C173 ¨C III¨C176.

function res = metricsQw(img1,img2,fused) 

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
        g = Qw(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
            g(k) = Qw(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = Qw(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end


function output = Qw(img1,img2,fused)

% function res=index_fusion(img1,img2,fuse,sw)
%
% This function is to calculate the fusion quality index proposed by
% im1, im2 -- input images;
% fused      -- fused image;
% res      -- metric value;
% sw       -- See the reference paper for three different outputs.
% 
% NOTE: ssim_index.m is needed. 
%
% Z. Liu @NRCC [4 Oct 2003]
%

% Ref: A new quality metric for image fusion, ICIP 2003
% by Piella et al. 
%

    s=size(size(img1));
    if s(2)==3 
        img1=rgb2gray(img1);
    else
        img1=img1;
    end 

    s1=size(size(img2));
    if s1(2)==3 
        img2=rgb2gray(img2);
    else
        img2=img2;
    end 
    
    s2=size(size(fused));
    if s2(2)==3 
        fused=rgb2gray(fused);
    else
        fused=fused;
    end 
    
    img1=double(img1);
    img2=double(img2);
    fused=double(fused);

[ssim,ssim_map,sigma1_sq,sigma2_sq]=ssim_index(img1,img2);
clear ssim, ssim_map;

buffer=sigma1_sq+sigma2_sq;
test=(buffer==0); test=test*0.5;
sigma1_sq=sigma1_sq+test; sigma2_sq=sigma2_sq+test;

buffer=sigma1_sq+sigma2_sq;
ramda=sigma1_sq./buffer;

[ssim1,ssim_map1]=ssim_index(fused,img1);
[ssim2,ssim_map2]=ssim_index(fused,img2);

sw = 3;

switch sw
    case 1
        Q=ramda.*ssim_map1+(1-ramda).*ssim_map2;
        %disp('fusion quality index: ');
        %mean2(Q)
%        res(1)=mean2(Q);
        res=mean2(Q);

    case 2
        
        % weighted fusion qualtiy index
        buffer(:,:,1)=sigma1_sq;
        buffer(:,:,2)=sigma2_sq;
        [Cw,U]=max(buffer,[],3);

        cw=Cw/sum(sum(Cw));
        Q=sum(sum(cw.*(ramda.*ssim_map1+(1-ramda).*ssim_map2)));
        %disp('weighted fusion quality index: ');
        %Q
%        res(2)=Q;
        res=Q;
% edge-dependent fusion quality index
    case 3
        flt1=[1 0 -1; 1 0 -1; 1 0 -1];
        flt2=[ 1 1 1; 0 0 0; -1 -1 -1];

        fuseX=filter2(flt1,fused,'same');
        fuseY=filter2(flt2,fused,'same');
        fuseF=sqrt(fuseX.*fuseX+fuseY.*fuseY);

        img1X=filter2(flt1,img1,'same');
        img1Y=filter2(flt2,img1,'same');
        img1F=sqrt(img1X.*img1X+img1Y.*img1Y);

        img2X=filter2(flt1,img2,'same');
        img2Y=filter2(flt2,img2,'same');
        img2F=sqrt(img2X.*img2X+img2Y.*img2Y);


        [ssim,ssim_map,sigma1_sq,sigma2_sq]=ssim_index(img1F,img2F);
        clear ssim, ssim_map;

        buffer=sigma1_sq+sigma2_sq;
        test=(buffer==0); test=test*0.5;
        sigma1_sq=sigma1_sq+test; sigma2_sq=sigma2_sq+test;

        buffer=sigma1_sq+sigma2_sq;
        ramda=sigma1_sq./buffer;

        [ssim1,ssim_map1]=ssim_index(fuseF,img1F);
        [ssim2,ssim_map2]=ssim_index(fuseF,img2F);

        buffer(:,:,1)=sigma1_sq;
        buffer(:,:,2)=sigma2_sq;
        [Cw,U]=max(buffer,[],3);

        cw=Cw/sum(sum(Cw));
        Qw=sum(sum(cw.*(ramda.*ssim_map1+(1-ramda).*ssim_map2)));

        alpha=1;
        %Qe=Q*Qw^alpha;
        Qe=Qw^alpha;
        %disp('edge-dependent fusion quality index: ');
        %Qe
%        res(3)=Qe;
        output=Qe;
end
end

function [mssim, ssim_map, sigma1_sq,sigma2_sq] = ssim_index(img1, img2, K, window, L)

%========================================================================
%SSIM Index, Version 1.0
%Copyright(c) 2003 Zhou Wang
%All Rights Reserved.
%
%The author is with Howard Hughes Medical Institute, and Laboratory
%for Computational Vision at Center for Neural Science and Courant
%Institute of Mathematical Sciences, New York University.
%
%----------------------------------------------------------------------
%Permission to use, copy, or modify this software and its documentation
%for educational and research purposes only and without fee is hereby
%granted, provided that this copyright notice and the original authors'
%names appear on all copies and supporting documentation. This program
%shall not be used, rewritten, or adapted as the basis of a commercial
%software or hardware product without first obtaining permission of the
%authors. The authors make no representations about the suitability of
%this software for any purpose. It is provided "as is" without express
%or implied warranty.
%----------------------------------------------------------------------
%
%This is an implementation of the algorithm for calculating the
%Structural SIMilarity (SSIM) index between two images. Please refer
%to the following paper:
%
%Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image
%quality assessment: From error measurement to structural similarity"
%IEEE Transactios on Image Processing, vol. 13, 2004, to appear.
%
%Kindly report any suggestions or corrections to zhouwang@ieee.org
%
%----------------------------------------------------------------------
%
%Input : (1) img1: the first image being compared
%        (2) img2: the second image being compared
%        (3) K: constants in the SSIM index formula (see the above
%            reference). defualt value: K = [0.01 0.03]
%        (4) window: local window for statistics (see the above
%            reference). default widnow is Gaussian given by
%            window = fspecial('gaussian', 11, 1.5);
%        (5) L: dynamic range of the images. default: L = 255
%
%Output: (1) mssim: the mean SSIM index value between 2 images.
%            If one of the images being compared is regarded as 
%            perfect quality, then mssim can be considered as the
%            quality measure of the other image.
%            If img1 = img2, then mssim = 1.
%        (2) ssim_map: the SSIM index map of the test image. The map
%            has a smaller size than the input images. The actual size:
%            size(img1) - size(window) + 1.
%
%Default Usage:
%   Given 2 test images img1 and img2, whose dynamic range is 0-255
%
%   [mssim ssim_map] = ssim_index(img1, img2);
%
%Advanced Usage:
%   User defined parameters. For example
%
%   K = [0.05 0.05];
%   window = ones(8);
%   L = 100;
%   [mssim ssim_map] = ssim_index(img1, img2, K, window, L);
%
%See the results:
%
%   mssim                        %Gives the mssim value
%   imshow(max(0, ssim_map).^4)  %Shows the SSIM index map
%
%========================================================================


if (nargin < 2 | nargin > 5)
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

if (size(img1) ~= size(img2))
   ssim_index = -Inf;
   ssim_map = -Inf;
   return;
end

[M N] = size(img1);

if (nargin == 2)
   if ((M < 11) | (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);	%
   K(1) = 0.01;								      % default settings
   K(2) = 0.03;								      %
   L = 255;                                  %
end

if (nargin == 3)
   if ((M < 11) | (N < 11))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   window = fspecial('gaussian', 11, 1.5);
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 4)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   L = 255;
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

if (nargin == 5)
   [H W] = size(window);
   if ((H*W) < 4 | (H > M) | (W > N))
	   ssim_index = -Inf;
	   ssim_map = -Inf;
      return
   end
   if (length(K) == 2)
      if (K(1) < 0 | K(2) < 0)
		   ssim_index = -Inf;
   		ssim_map = -Inf;
	   	return;
      end
   else
	   ssim_index = -Inf;
   	ssim_map = -Inf;
	   return;
   end
end

C1 = (K(1)*L)^2;
C2 = (K(2)*L)^2;
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