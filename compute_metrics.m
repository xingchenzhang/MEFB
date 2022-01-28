% This code can be used to compute values of selected metrics for selected
% algorithms on selected image pairs
%
% Note: Please change the path in line 26 to your own path before running
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

addpath('./metrics');
addpath('./util');
addpath('./methods');

path = 'Your own path to MEFB\output';
fusedPath = [path '\fused_images\'];
outputPath = [path '\evaluation_metrics\'];
outputPathSingle = [path '\evaluation_metrics_single\'];

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

if ~exist(outputPathSingle,'dir')
    mkdir(outputPathSingle);
end

imgs = configImgs;

methods = configMethods;
metrics = configMetrics;

numImgs=length(imgs);
numMethods=length(methods);
numMetrics=length(metrics);

% output information
fid=fopen(strcat(path, '\information.txt'),'w');
fprintf(fid,'%15s\r\n','The image paris are:');
for i=1:numImgs
    fprintf(fid ,'%15s\r\n',imgs{i}.name);
end

fprintf(fid,'%15s\r\n','');
fprintf(fid,'%15s\r\n','The methods are:');
for i=1:numMethods
    fprintf(fid,'%15s\r\n', methods{i}.name);
end

fprintf(fid,'%15s\r\n','');
fprintf(fid,'%15s\r\n','The metrics are:');
for i=1:numMetrics
    fprintf(fid,'%15s\r\n', metrics{i}.name);
end
fclose(fid);

Asualization = 0;
resultsMetrics = zeros(numImgs, numMethods, numMetrics);
            
for idxMethod=1:numMethods
    m = methods{idxMethod};

    for idxImgs=1:length(imgs)
        s = imgs{idxImgs};

        sA.img = strcat(s.path,s.name, '_A.',s.ext);
        sB.img = strcat(s.path,s.name, '_B.',s.ext);

        imgA = imread(sA.img);
        imgB = imread(sB.img);

        [imgH_A,imgW_A,chA] = size(imgA);
        [imgH_B,imgW_B,chB] = size(imgB);
        
        for idxMetrics = 1:numMetrics
            
            sMetrics = metrics{idxMetrics};
        
            fusedName = [fusedPath s.name '_' m.name '.' s.ext];
            if exist([fusedPath s.name '_' m.name '.' s.ext])
                sFused = imread(fusedName);              
                % check whether the result exists
                if exist(strcat(outputPathSingle,s.name, '_', m.name,'_',sMetrics.name ,'.txt'))    
                    A=importdata(strcat(outputPathSingle,s.name, '_', m.name,'_',sMetrics.name ,'.txt'));              
                    resultsMetrics(idxImgs, idxMethod, idxMetrics) = A;
                    continue;
                end
                
                disp([num2str(idxMethod) '_' m.name ', ' num2str(idxImgs) '_' s.name ', ' num2str(idxMetrics) '_' sMetrics.name])       

                funcName = ['res = metrics' sMetrics.name '(imgA, imgB, sFused);'];
                disp(funcName)

                try
                    cd(['./metrics/']);
                    addpath(genpath('./'))

                    eval(funcName);

                catch err
                    disp('error');
                    rmpath(genpath('./'))
                    cd('../../')
                    continue;
                end
                
                resultsMetrics(idxImgs, idxMethod, idxMetrics) = res;
                
                outputFileSingle = strcat(outputPathSingle,s.name, '_', m.name,'_',sMetrics.name ,'.txt');

                dlmwrite(outputFileSingle,res)

                cd('../');
                
            else                
               str=['The fused image ' fusedName ' does not exists, please check'];
               disp(str)           
            end            
        end               
    end  
end

outputFile = strcat(outputPath, 'evaluationMetrics.mat');
save(outputFile,'resultsMetrics');

% compute the average value of each metric on all image paBs
resultsMetricsAverageImg = nanmean(resultsMetrics,1); 
outputFileAverage = strcat(outputPath, 'evaluationMetricsAverageImg.mat');
save(outputFileAverage,'resultsMetricsAverageImg');
