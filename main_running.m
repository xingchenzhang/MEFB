% This is the main program of MEFB and can be used to produce fused images.
%
% Note: Please change the path in line 24 to your own path before running
%
% If you use this code, please site the following paper:
%
% Zhang, Xingchen. "Benchmarking and comparing multi-exposure image fusion algorithms." 
% Information Fusion (2021).
%
% Thanks a lot!
%
% For more information, please see https://github.com/xingchenzhang/MEFB
%
% Contact: xingchen.zhang@imperial.ac.uk

close all
clear
clc
warning off all;

addpath('./util');

addpath(['./methods'])
path = 'Your own path\MEFB\output';
outputPath = [path '\fused_images\'];

imgs = configImgs;

methods=configMethods;

numImgs=length(imgs);
numMethods=length(methods);

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

visualization = 0;
            
for idxMethod=1:numMethods
    m = methods{idxMethod};
    t1 = clock;

    j =0;
    for idxImgs=1:length(imgs)
        s = imgs{idxImgs};

        sA.img = strcat(s.path,s.name, '_A.',s.ext);
        sB.img = strcat(s.path,s.name, '_B.',s.ext);
        
        sA.ext = s.ext;
        SB.ext = s.ext;
        
        sA.name = s.name;
        sB.name = s.name;

        imgA = imread(sA.img);
        imgB = imread(sB.img);

        [imgH_A,imgW_A,chA] = size(imgA);
        [imgH_B,imgW_B,chB] = size(imgB);
        
        % check whether the result exists
        if exist([outputPath s.name '_' m.name '.' s.ext])
            continue;
        end
        
        disp([num2str(idxMethod) '_' m.name ', ' num2str(idxImgs) '_' s.name])       

        funcName = ['img = run_' m.name '(sA, sB, outputPath, m, visualization);'];

        try

            cd(['./methods/' m.name]);
            addpath(genpath('./'))
            
            eval(funcName);
            j=j+1;
            
        catch err
            disp('error');
            rmpath(genpath('./'))
            cd('../../')
            continue;
        end
        
        imwrite(img, [outputPath '/' s.name '_' m.name '.' s.ext]);
        cd('../../');
    end
    
    t2=clock;
    runtimeAverage = etime(t2,t1)./j;
        
    str=['The total runtime of ' m.name ' is: ' num2str(etime(t2,t1)) 's'];
    disp(str)
    
    str=['The average runtime of ' m.name ' per image is: ' num2str(runtimeAverage) 's'];
    disp(str)
end