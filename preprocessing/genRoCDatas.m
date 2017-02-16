% for scan num5 generate all voxel patch-wised dataset
clear ; close all; clc
cropSizeX = 25;
cropSizeY = 25;
cropSizeZ = 25;

strideX = 1;
strideY = 1;
strideZ = 1;

sourceImgId = 'all_testimg_1127_fcnn';
sourceLabelId = 'all_testlabel_1127_fcnn';
testimgPath = strcat('results/',sourceImgId,'.mat');
testData = load(testimgPath);
testData = testData.testData;
testlabelPath = strcat('results/',sourceLabelId,'.mat');
testLabel = load(testlabelPath);
testLabel = testLabel.teLabel;

testData1 = squeeze(testData(9,:,:,:,:));
testLabel1 = squeeze(testLabel(9,:,:,:,:));


[imgZ,imgY,imgX] = size(testData1);
            
maxStepsX = floor((imgX-cropSizeX)/strideX)+1;
maxStepsY = floor((imgY-cropSizeY)/strideY)+1;

% 488*488  98*98=9604
maxStep = maxStepsY*maxStepsX;

cropImages = uint8(zeros(maxStep,cropSizeZ,cropSizeY,cropSizeX));


Idx = 1;
    zMiddle = 17;
    zStart = zMiddle-12;
    zEnd = zMiddle+12;
            
    for stepY=1:maxStepsY
        for stepX=1:maxStepsX
            yStart = (stepY-1)*strideY+1;
            yEnd = (stepY-1)*strideY+cropSizeY;
            xStart = (stepX-1)*strideX+1;
            xEnd = (stepX-1)*strideX+cropSizeX;
            yMiddle = floor((yStart+yEnd)/2);
            xMiddle = floor((xStart+xEnd)/2);

            voxelImage = testData1(zStart:zEnd,yStart:yEnd,xStart:xEnd);
            voxelSeg = testLabel1(zStart:zEnd,...
            max(yStart,1):min(yEnd,512),...
            max(xStart,1):min(xEnd,512));

            % if voxelImage is not black then add it to repos.
            % todo: if we need round the seg?
            isStroke = testLabel1(zMiddle,yMiddle,xMiddle);
            if(isStroke>0)
                fprintf('******find STROKE at: z:%d,y:%d,x:%d*****\n',zMiddle,yMiddle,xMiddle);
                cropImages(Idx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                Idx = Idx+1;

            else
                cropImages(Idx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                Idx = Idx+1;
            end

        end
        
    end


cropImages(Idx:end,:,:,:) = [];


save(strcat('results/test/',sourceImgId,'_rocSourceData.mat'),'cropImages','-v7.3')






