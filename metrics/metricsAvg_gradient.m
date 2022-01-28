% The source code is from the authors of the metric
% The interface is modified by the author of MEFB to integrate it into MEFB.
%
% Reference for the metric:
% G. Cui, H. Feng, Z. Xu, Q. Li, and Y. Chen, ?¡ãDetail preserved fusion of visible and infrared images using regional
% saliency extraction and multi-scale image decomposition,?¡À Optics Communications, vol. 341, pp. 199 ¡§C 209, 2015

function res = metricsAvg_gradient(img1,img2,fused) 
 
if nargin == 3 
    fused = double(fused); 
    % Get the size of img 
    [r,c,b] = size(fused); 
    [m1,n1,b1] = size(img1);
    [m2,n2,b2] = size(img2);
    
    if (b1 == 1) && (b2 ==1) && (b == 3)
        fused = rgb2gray(fused);
    end
    [r,c,b] = size(fused); 
     
    dx = 1; 
    dy = 1; 
    for k = 1 : b 
        band = fused(:,:,k); 
        [dzdx,dzdy] = gradient(band,dx,dy); 
        s = sqrt((dzdx .^ 2 + dzdy .^2) ./ 2); 
        g(k) = sum(sum(s)) / ((r - 1) * (c - 1)); 
    end 
    res = mean(g); 
else 
    error('Wrong number of input!'); 
end

 