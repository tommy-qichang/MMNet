function storeDataVoxelwised(i)
    savePrefix=strcat('0112_25^3_',i);
    outputPrefix = strcat('0113_25^3_',i);
%     cropSizeX = 50;
%     cropSizeY = 50;
%     cropSizeZ = 10; 
    cropSizeX = 25;
    cropSizeY = 25;
    cropSizeZ = 25; 
    strideX = 5;%512->200 10->
    strideY = 5;
    strideZ = 1;
    
%     negativeMarginSize = 50;
    negativeMarginSize = 30;
    
    
    trDataPath = strcat('clean_data/all_img_' , savePrefix);
    trLabelPath = strcat('clean_data/all_label_' , savePrefix);
    teDataPath = strcat('clean_data/all_testimg_' , savePrefix);
    teLabelPath = strcat('clean_data/all_testlabel_' , savePrefix);

    fprintf(strcat('begin loading train data and label...',trDataPath,'\n'));
    trData = load(trDataPath);
    trLabel = load(trLabelPath);
    trData = trData.trainData;
    trLabel = trLabel.trLabel;
    fprintf('finish loading train data and label...\n');
    
%     trData = trData(:,8:end-8,:,:);
%     trLabel = trLabel(:,8:end-8,:,:);
    
    
    tranverseAllImages(trData,trLabel,'train');
    
    
    fprintf(strcat('begin loading test data and label...',teDataPath,'\n'));
    teData = load(teDataPath);
    teLabel = load(teLabelPath);
    teData = teData.testData;
    teLabel = teLabel.teLabel;
    fprintf('finish loading test data and label...\n');
    
    teData = teData(:,:,:,:);
    teLabel = teLabel(:,:,:,:);
    
    
    tranverseAllImages(teData,teLabel,'test');

    
    
    function tranverseAllImages(imgData,labelData,dataPrefix)
        allCropImagesPos = uint8(zeros(0,cropSizeZ,cropSizeY,cropSizeX));
        allCropImagesNeg = uint8(zeros(0,cropSizeZ,cropSizeY,cropSizeX));

        for i=1:size(imgData,1)
            dcm = squeeze(imgData(i,:,:,:));
            seg = squeeze(labelData(i,:,:,:));
            [imgZ,imgY,imgX] = size(dcm);
            
            maxStepsX = floor((imgX-cropSizeX)/strideX)+1;
            maxStepsY = floor((imgY-cropSizeY)/strideY)+1;
            maxStepsZ = floor((imgZ-cropSizeZ)/strideZ)+1;
            
            maxStep = maxStepsZ*maxStepsY*maxStepsX;
            cropImagesPos = uint8(zeros(maxStep,cropSizeZ,cropSizeY,cropSizeX));
            cropImagesNeg = uint8(zeros(maxStep,cropSizeZ,cropSizeY,cropSizeX));

            posIdx = 1;
            negIdx = 1;
            for stepZ=1:maxStepsZ
                
                zStart = (stepZ-1)*strideZ+1;
                zEnd = (stepZ-1)*strideZ+cropSizeZ;
                zMiddle = floor((zStart+zEnd)/2);
                for stepY=1:maxStepsY
                    for stepX=1:maxStepsX
                        if(stepX==1&&stepY==1&&stepZ==1)
                            fprintf('maxStepX:%d;maxStepY:%d;maxStepZ:%d \n',maxStepsX,maxStepsY,maxStepsZ)
                        end
                            
%                         fprintf('x:%d; y:%d; z:%d \n',stepX,stepY,stepZ);
                        yStart = (stepY-1)*strideY+1;
                        yEnd = (stepY-1)*strideY+cropSizeY;
                        xStart = (stepX-1)*strideX+1;
                        xEnd = (stepX-1)*strideX+cropSizeX;
                        yMiddle = floor((yStart+yEnd)/2);
                        xMiddle = floor((xStart+xEnd)/2);
                        
                        voxelImage = dcm(zStart:zEnd,yStart:yEnd,xStart:xEnd);
                        voxelSeg = seg(zStart:zEnd,...
                            max(yStart-negativeMarginSize,1):min(yEnd+negativeMarginSize,200),...
                            max(xStart-negativeMarginSize,1):min(xEnd+negativeMarginSize,200));
%                         if(sum(abs(voxelImage(:)))>0 && sum(abs(voxelSeg(:)))>0)
                        if(dcm(zMiddle,yMiddle,xMiddle)>0 && sum(abs(voxelSeg(:)))>0)
                            % if voxelImage is not black then add it to repos.
                            % todo: if we need round the seg?
                            % isStroke = seg(zMiddle,yMiddle,xMiddle);
                            % if the stroke in voxel area, then set as
                            % positive.
                            isStroke = seg(zMiddle,yMiddle,xMiddle);
                            
                            if(isStroke>0)
                                fprintf('******find STROKE at: z:%d,y:%d,x:%d*****\n',zMiddle,yMiddle,xMiddle);
                                cropImagesPos(posIdx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                                posIdx = posIdx+1;
                            else
                                cropImagesNeg(negIdx,:,:,:) = reshape(voxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                                negIdx = negIdx+1;
                            end
                            
                            
                        end
                        
%                         currentStep = (stepZ-1)*(maxStepsX*maxStepsY)+(stepY-1)*(maxStepsX)+stepX;
%                         if rem(currentStep,10)==0
%                             fprintf('%.3f stepz:%d stepy:%d stepx:%d finished... \n',(currentStep/maxStep),stepZ,stepY,stepX);
%                         end
                    end
                end
                
                % add positive cases if this slide has stroke. In case
                % ignore the small part of stroke.
                slideSeg = squeeze(seg(zMiddle,:,:));
                
                if(sum(slideSeg(:))>2)
                    % has stroke 34 middleIndex:20
                    fprintf('have stroke,middleZ:%d, add first one and last one into the repos.\n',zMiddle);
                    findIdx = find(slideSeg==2);
                    if(size(findIdx,1)==0)
                        continue;
                    end
                    % first one:
                    firstIdx = findIdx(1);
                    firstSegX = floor(firstIdx/imgY);
                    firstSegStartX = round(max((firstSegX-(cropSizeX/2)),1));
                    firstSegEndX = firstSegStartX+cropSizeX-1;
                    firstSegY = rem(firstIdx,imgY);
                    firstSegStartY = round(max((firstSegY-(cropSizeY/2)),1));
                    firstSegEndY = firstSegStartY+cropSizeY-1;
                    
                    if((firstSegX-(cropSizeX/2))<0 || (firstSegY-(cropSizeY/2))<0)
                        fprintf('*****Maybe Wrong Segmentation?******');
                        continue;
                    end
                    
                    firstVoxelImage = dcm(zStart:zEnd,firstSegStartY:firstSegEndY...
                        ,firstSegStartX:firstSegEndX);
                    cropImagesPos(posIdx,:,:,:) = reshape(firstVoxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                    posIdx = posIdx+1;
                    
                    % last one;
                    lastIdx = findIdx(end);
                    lastSegX = floor(lastIdx/imgY);
                    lastSegStartX = round(max((lastSegX-(cropSizeX/2)),1));
                    lastSegEndX = lastSegStartX+cropSizeX-1;
                    
                    lastSegY = rem(lastIdx,imgY);
                    lastSegStartY = round(max((lastSegY-(cropSizeY/2)),1));
                    lastSegEndY = lastSegStartY+cropSizeY-1;
                    
                    lastVoxelImage = dcm(zStart:zEnd,lastSegStartY:lastSegEndY...
                        ,lastSegStartX:lastSegEndX);
                    cropImagesPos(posIdx,:,:,:) = reshape(lastVoxelImage,1,cropSizeZ,cropSizeY,cropSizeX);
                    posIdx = posIdx+1;
                    
                end
                
                
                fprintf('current Z index:%d\n',stepZ);
            end
            
            cropImagesPos(posIdx:end,:,:,:) = [];
            cropImagesNeg(negIdx:end,:,:,:) = [];
            
            fprintf('===finish scan id: %d, increasePos:%d, increaseNeg:%d ...===\n'...
                ,i,size(cropImagesPos,1),size(cropImagesNeg,1));
            
            
            allCropImagesPos = [allCropImagesPos;cropImagesPos];
            allCropImagesNeg = [allCropImagesNeg;cropImagesNeg];
            
                
%             if(rem(i,50)==0 || i==size(imgData,1))
%                 saveIdx = ceil(i/50);
%                 save(strcat('results/patch_',dataPrefix,'_pos_',num2str(saveIdx),'_',outputPrefix,'.mat'),'allCropImagesPos','-v7.3');
%                 save(strcat('results/patch_',dataPrefix,'_neg_',num2str(saveIdx),'_',outputPrefix,'.mat'),'allCropImagesNeg','-v7.3');
%                 fprintf('save result for save index: %d \n',saveIdx);
%                 
%                 allCropImagesPos = uint8(zeros(0,cropSizeZ,cropSizeY,cropSizeX));
%                 allCropImagesNeg = uint8(zeros(0,cropSizeZ,cropSizeY,cropSizeX));
%             end
            
        end
        
        save(strcat('clean_data/patch_',dataPrefix,'_pos_',outputPrefix,'.mat'),'allCropImagesPos','-v7.3');
        save(strcat('clean_data/patch_',dataPrefix,'_neg_',outputPrefix,'.mat'),'allCropImagesNeg','-v7.3');
        fprintf('*****save result for save **** \n');
        
        
    end

end

