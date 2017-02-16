clear ; close all; clc
rootPath = '../training/25^3_fcnn_i19_vnet/rocData/';
%i13
% Iter:7,DICE: 9.79 
% Iter:11,DICE: 21.66 
% Iter:15,DICE: 29.77 
% Iter:19,DICE: 18.66 
% Iter:23,DICE: 13.84 
% Iter:27,DICE: 17.36 
% Iter:31,DICE: 16.58 
% Iter:35,DICE: 19.03 
% Iter:39,DICE: 28.63 

%%% 
% Iter:7,DICE: 16.67 
% Iter:11,DICE: 33.34 
% Iter:15,DICE: 38.50 
% Iter:19,DICE: 28.56 
% Iter:23,DICE: 26.76 
% Iter:27,DICE: 28.52 
% Iter:31,DICE: 29.45 
% Iter:35,DICE: 31.25 
% Iter:39,DICE: 37.21 
% Iter:43,DICE: 39.13 

%%%43-pretrained - i15
% Iter:3,DICE: 29.82 
% Iter:7,DICE: 26.49 
% Iter:11,DICE: 22.22 
% Iter:15,DICE: 33.98 
% Iter:19,DICE: 41.38  ------->current best
% Iter:23,DICE: 36.31 

%% i16 fix duplicate test case problem.
% Iter:3,DICE: 36.58 
% Iter:7,DICE: 32.53 
% Iter:11,DICE: 12.69 
% Iter:15,DICE: 22.60 
% Iter:19,DICE: 14.02 

%i17 
% Iter:7,DICE: 17.38 
% Iter:11,DICE: 13.52 
% Iter:15,DICE: 15.87 
% Iter:19,DICE: 14.84 
% Iter:23,DICE: 13.81 
% Iter:27,DICE: 19.97 
% Iter:31,DICE: 14.34 
% Iter:35,DICE: 11.22 

%i18 retrainif modelPath and paths.filep(modelPath) then
    model:add(torch.load(modelPath));
    print('==> load exist model:' .. modelPath);
else
    model:add(dofile(opt.model .. '.lua'):cuda());
end


% Iter:3,DICE: 0.09 
% Iter:7,DICE: 13.46 
% Iter:11,DICE: 14.67 
% Iter:15,DICE: 20.78 
% Iter:19,DICE: 8.89 
% Iter:23,DICE: 27.25 
% Iter:27,DICE: 20.21 
% Iter:31,DICE: 27.51 
% Iter:35,DICE: 28.48 
% Iter:39,DICE: 22.10 
% Iter:43,DICE: 26.95 
% Iter:47,DICE: 32.70 
% Iter:51,DICE: 24.50 
% Iter:55,DICE: 26.50 

% i19 vnet
% Iter:2,DICE: 5.91 
% Iter:6,DICE: 1.14 
% Iter:10,DICE: 32.95 
% Iter:14,DICE: 32.51 
% Iter:18,DICE: 1.26 
% Iter:22,DICE: 21.19 
% Iter:26,DICE: 21.67 
% Iter:30,DICE: 34.69 
% Iter:34,DICE: 34.43 
% Iter:38,DICE: 32.00 
% Iter:42,DICE: 44.84 
% Iter:46,DICE: 19.34 

startId = 30;
endId = 46;
%threshold = 0.9;
threshold = 0;

for id = startId:4:endId
    resultid = id;
arrAllTimeExp = zeros(listLen,1);
arrAllTimeSlow = zeros(listLen,1);

    
    result = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/result');
    target = hdf5read(strcat(rootPath,'fcnn_rocdata_epo_result',num2str(resultid),'.h5'),'/target');

    result = permute(result,[3,2,1]);
    target = permute(target,[3,2,1]);
    target = target -1;
    
    
    result(result>=threshold)=1;
    result(result<threshold)=0;
    [b,x,y] = size(result);
    
    flatTarget = reshape(target,[b,x*y]);
    flatTarget = sum(flatTarget,2);
    flatTarget(flatTarget>0)=1;
    flatResult = reshape(result,[b,x*y]);
    flatResult(flatTarget==0,:)=0;
    result = reshape(flatResult,[b,x,y]);
    
%     for i=1:lLen
%         slideTarget = squeeze(target(i,:,:));
%         slideResult = squeeze(result(i,:,:));
%         if sum(slideTarget(:))==0
%             result(i,:,:) = 0;
%         end
%     end
%     andOp = (result&target)*2;
%     orOp = result|target;
%     dice = sum(andOp(:))/sum(orOp(:));

     common = (target & result);
     a = sum(common(:));
     b = sum(target (:));
     c = sum(result(:));
     dice = 2*a/(b+c);
   
%      common = (target & result);
%      join = target | result;
%      a = sum(common(:));
%      b = sum(join(:));
%      dice = 2*a/(b);
     
     
%      common = dot(target(:),result(:));
%      b = dot(target(:),target(:));
%      c = dot(result(:),result(:));
%      
%      dice = 2*common/(b+c);
     
     
    fprintf(strcat('Iter:',num2str(id),',DICE: %.2f \n'),dice*100);
    
end

