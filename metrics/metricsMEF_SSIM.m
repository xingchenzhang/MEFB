% The source code is from the authors of the metric (https://kedema.org/Publications.html)
% The interface is modified by the author of MEFB to integrate it into MEFB. 
% 
% Reference for the metric:
% K. Ma, K. Zeng, Z. Wang, Perceptual quality assessment for multi-exposure image fusion, IEEE Trans. Image 
% Process. 24 (11) (2015) 3345¨C3356.

function res= metricsMEF_SSIM(img1,img2,fused)
    
    imgSeqColor(:,:,:,1) = img1;
    imgSeqColor(:,:,:,2) = img2;
    
    [s1, s2, s3, s4] = size(imgSeqColor);
    imgSeq = zeros(s1, s2, s4);
    for i = 1:s4
        imgSeq(:, :, i) =  rgb2gray( squeeze( imgSeqColor(:,:,:,i) ) ); % color to gray conversion
    end
    fI1 = fused; 
    fI1 = double(rgb2gray(fI1));
    [Q(1), Qs1, QMap1] = mef_ms_ssim(imgSeq, fI1);
    res = Q;
    
end
        