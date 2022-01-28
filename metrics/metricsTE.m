% The source code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/metricMI.m
% The interface is modified by the author of MEFB to integrate it into MEFB. 
% 
% Reference for the metric:
% N. Cvejic, C. Canagarajah, D. Bull, Image fusion metric based on mutual 
% information and tsallis entropy, Electron. Lett. 42 (11) (2006) 626¨C627.

function res = metricsTE(img1,img2,fused) 

%     fused = double(fused); 
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

%     img1 = double(img1);
%     img2 = double(img2);

    if b == 1
        g = TE(img1,img2,fused);
        res = g;
    elseif b1 == 1
        for k = 1 : b 
            g(k) = TE(img1(:,:,k),img2,fused(:,:,k)); 
        end 
        res = mean(g); 
    else
        for k = 1 : b 
            g(k) = TE(img1(:,:,k),img2(:,:,k),fused(:,:,k)); 
        end 
        res = mean(g); 
    end
end


function output = TE(im1,im2,fused)

% function res=metricMI(im1,im2,fused,sw)
%
% This function implements the revised mutual information algorithms for fusion metric.
% im1, im2 -- input images;
% fused      -- fused image;
% sw       -- 1: revised MI; 2: Tsallis entropy (Cvejie); 3: Nava.
% res      -- metric value;
%
% IMPORTANT: The size of the images need to be 2X. This is not an
% implementation of Qu's algorithm. See the function for details.
%
% Z. Liu [July 2009]
%

% Ref: Comments on "Information measure for performance of image fusion"
% By M. Hossny et al.
% Electronics Letters Vol. 44, No.18, 2008
%
% Ref: Mutual information impoves image fusion quality assessments
% By Rodrigo Nava et al.
% SPIE Newsroom
%
% Ref: Image fusion metric based on mutual information and Tsallis entropy
% By N. Cvejie et al.
% Electronics Letters, Vol.42, No. 11, 2006

%% pre-processing
im1=normalize1(im1);
im2=normalize1(im2);
fused=normalize1(fused);

% if nargin==3
%     sw=1;
% end

%switch sw
%     case 1
%         % revised MI algorithm (Hossny)
%         [I_fx,H_xf,H_x,H_f1]=mutual_info(im1,fused);
%         [I_fy,H_yf,H_y,H_f2]=mutual_info(im2,fused);
%         
%         MI=2*(I_fx/(H_f1+H_x)+I_fy/(H_f2+H_y));
%         res=MI;
%    case 2
        q=1.85;    % Cvejic's constant
        I_fx=tsallis(im1,fused,q);
        I_fy=tsallis(im2,fused,q);
        output=I_fx+I_fy;
        
%     case 3
%         % MI and Tsallis entropy
%         % set up constant q
%         q=0.43137; % Nava's constant
%         
%         I_fx=tsallis(im1,fused,q);
%         I_fy=tsallis(im2,fused,q);
%         I_xy=tsallis(im1,im2,q);        
%        
%         [M_xy,H_xy,H_x,H_y]=mutual_info(im1,im2);
% 
%         MI=(I_fx+I_fy)/(H_x.^q+H_y.^q+I_xy);
%         res=MI;
%     otherwise
%         error('Your input is wrong! Please check help file.');
%end
end

function RES=tsallis(im1,im2,q)

    % function RES=tsallis(im1,im2,q)
    % 
    % This function is caculate Tsallis entropy for two input images.
    % im1   -- input image one;
    % im2   -- input image two;
    % q     -- constant
    %
    % RES    -- Tsallis entropy;
    %
    %
    % Note: The input images need to be in the range of 0-255. 
    %
    % Z. Liu @ NRCC [July 17, 2009]

    im1=double(im1);
    im2=double(im2);

    [hang,lie]=size(im1);
    count=hang*lie;
    N=256;

    %% caculate the joint histogram
    h=zeros(N,N);

    for i=1:hang
        for j=1:lie
            % in this case im1->x, im2->y
            h(im1(i,j)+1,im2(i,j)+1)=h(im1(i,j)+1,im2(i,j)+1)+1;
        end
    end

    %% marginal histogram

    % this operation converts histogram to probability
    h=h./sum(h(:));

    im1_marg=sum(h);
    im2_marg=sum(h');

    result=0;
    for i=1:N
        for j=1:N
            buff=im1_marg(i)*im2_marg(j);
            if buff~=0
                result=result+h(i,j).^q/(buff).^(q-1);
            end
        end
    end

    RES=(1-result)/(1-q);
end


