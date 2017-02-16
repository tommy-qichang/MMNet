prefix = '1121_deskull_60_5';

allImgDataPath = strcat('results/all_img_',prefix,'.mat');
allImgLabelPath = strcat('results/all_label_',prefix,'.mat');
testImgDataPath = strcat('results/all_testimg_',prefix,'.mat');
testImgLabelPath = strcat('results/all_testlabel_',prefix,'.mat');

testPath = 'backup/tmp/';
extentPadding = 200;

allImageData = load(allImgDataPath);
allImageData = allImageData.trainData;
allImageLabel = load(allImgLabelPath);
allImageLabel = allImageLabel.trLabel;
finalHeight = 0;
finalWidth = 0;
for i=1:size(allImageData,1)
    imageData = squeeze(allImageData(i,:,:,:));
    [z,y,x] = size(imageData);
    imageLabel = squeeze(allImageLabel(i,:,:,:));

    [center,firstY,firstX] = ind2sub(size(imageData),find(imageData >4));
    coord = [center,firstY,firstX];
    low = min(coord);
    high = max(coord);
    center = round((high + low)/2);
    
    for j=1:z
        curImg = imageData(j,center(2)-extentPadding:center(2)+extentPadding,center(3)-extentPadding:center(3)+extentPadding);
        
        imwrite(squeeze(curImg),strcat(testPath,num2str(i),'_',num2str(j),'.jpg'));
    end
    
    
%     centerImg = squeeze(imageData(r(1),:,:));
%     [Y,X] = ind2sub(size(squeeze(centerImg)),find(centerImg >4));
%     coord = [Y,X];
%     lu = round(min(coord)-0.5);
%     rl = round(max(coord)+0.5);
%     resultimg = centerImg(lu(1):rl(1),lu(2):rl(2));
%     height = rl(1)-lu(1);
%     width = rl(2) - lu(2);
%     if height > finalHeight
%         finalHeight = height;
%     end
%     if width > finalWidth
%         finalWidth = width;
%     end
%     imwrite(squeeze(result),strcat('backup/tmp/',num2str(i),'.jpg'));
%     imwrite(squeeze(result2),strcat('backup/tmp/',num2str(i),'-2.jpg'));
%     fprintf('finalHeight:%d,finalWidth%d idx:%d \n',finalHeight,finalWidth,i);
%     
end




