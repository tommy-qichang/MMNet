clear ; close all; clc
outputPrefix = '50^2+10';

for i=1:4
    
    patchNegPath = strcat('results/patch_train_neg_',num2str(i),'_',outputPrefix,'.mat');
    trNegData = load(patchNegPath);
    trNegData = trNegData.allCropImagesNeg;
    
    
    patchPosPath = strcat('results/patch_train_pos_',num2str(i),'_',outputPrefix,'.mat');
    trPosData = load(patchPosPath);
    trPosData = trPosData.allCropImagesPos;
    
    if exist('trAllNegData','var')
        trAllNegData = [trAllNegData;trNegData];
        trAllPosData = [trAllPosData;trPosData];
    else
        trAllNegData = trNegData;
        trAllPosData = trPosData;
    end
    
    fprintf('finish training data parsing:%d\n',i)
end

save(strcat('results/patch_train_neg_',outputPrefix,'.mat'),'trAllNegData','-v7.3');

save(strcat('results/patch_train_pos_',outputPrefix,'.mat'),'trAllPosData','-v7.3');


patchNegPath = strcat('results/patch_test_neg_1_',outputPrefix,'.mat');
teNegData = load(patchNegPath);
teNegData = teNegData.allCropImagesNeg;


patchPosPath = strcat('results/patch_test_pos_1_',outputPrefix,'.mat');
tePosData = load(patchPosPath);
tePosData = tePosData.allCropImagesPos;

teAllNegData = teNegData;
teAllPosData = tePosData;

fprintf('finish test data parsing\n');
 

save(strcat('results/patch_test_neg_',outputPrefix,'.mat'),'teAllNegData','-v7.3');

save(strcat('results/patch_test_pos_',outputPrefix,'.mat'),'teAllPosData','-v7.3');



