% This code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/mutual_info.m

function [MI,H_xy,H_x,H_y]=mutual_info(im1,im2)

% function [MI,H_xy,H_x,H_y]=mutual_info(im1,im2)
% 
% This function is caculate the mutual information of two input images.
% im1   -- input image one;
% im2   -- input image two;
%
% MI    -- mutual information;
%
%
% Note: The input images need to be in the range of 0-255. (see function:
% normalize1.m)
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

H_x=0;   % for im1
H_y=0;   % for im2

%for i=1:count
%for i=1:N
%    if (im1_marg(i)>eps)
%        % entropy for image1
%        H_x=H_x+(-im1_marg(i)*(log2(im1_marg(i))));
%    end
%    if (im2_marg(i)>eps)
%        % entropy for image2
%        H_y=H_y+(-im2_marg(i)*(log2(im2_marg(i))));
%    end
%end

H_x=-sum(im1_marg.*log2(im1_marg+(im1_marg==0)));
H_y=-sum(im2_marg.*log2(im2_marg+(im2_marg==0)));


% joint entropy
H_xy=-sum(sum(h.*log2(h+(h==0))));

% mutual information
MI=H_x+H_y-H_xy;
