function predictionGen()

sourcePath = '25^3_i8_r0.005';
scanPath = strcat('results/training/',sourcePath,'/rocData/fcnn_rocdata_epo_result*.mat');
results = dir(char(scanPath));
resultsNum = size(results,1);

targetPath = strcat('results/training/',sourcePath,'/rocData/fcnn_rocdata_epo_target*.mat');
targets = dir(char(targetPath));

for i=1:resultsNum
    resultPath = strcat('results/training/',sourcePath,'/rocData/',results(i).name);
    targetPath = strcat('results/training/',sourcePath,'/rocData/',targets(i).name);
    result1 = load(resultPath);
    target1 = load(targetPath);
    

end
predictionGen();