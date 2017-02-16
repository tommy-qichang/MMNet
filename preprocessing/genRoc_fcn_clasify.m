clear ; close all; clc
% rootPath = '../training/25^3_fcnn_i12_r0.1_w150/rocData/';
% rootPath = '../training/25^3_fcnn_i14_r0.1_w50/rocData/';
% rootPath = '../training/25^3_fcnn_i19_vnet/rocData/';
rootPath = '../training/25^3_i20_classify/rocData/';

% % i20 classify
% iter:5 with AUC:0.9236 
% iter:9 with AUC:0.9252 
% iter:13 with AUC:0.9237 
% iter:17 with AUC:0.9121 


startId =9;
endId = 9;

for id = startId:4:endId
    resultid = id;
    
    result = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/result');
    target = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/target');

    result = permute(result,[2,1]);
    finalResult = result(:,2)-result(:,1);
%     target = permute(target,[3,2,1]);


    [X,Y,T,AUC] = perfcurve(target,finalResult,2);

    fprintf(strcat('iter:',num2str(id),' with AUC:%.4f \n'),AUC);
end






