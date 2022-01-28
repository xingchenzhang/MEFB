% This code is from https://github.com/zhengliu6699/imageFusionMetrics/blob/master/normalize1.m

function RES=normalize1(data)

% function RES=normalize1(data)
%
% This function is to NORMALIZE the data. 
% The data will be in the interval 0-255 (gray level) and pixel value has
% been rounded to an integer.
% 
% See also: normalize.m 
%
% Z. Liu @NRCC (Aug 24, 2009)

data=double(data);
da=max(data(:));
xiao=min(data(:));
if (da==0 & xiao==0)
    RES=data;
else
    newdata=(data-xiao)/(da-xiao);
    RES=round(newdata*255);
end


