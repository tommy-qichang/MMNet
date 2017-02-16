function storeScansAllInOne2()
    clear ; close all; clc
    imageRootPath = 'images';
    annotationPath = 'annotations';
    savePrefix='1205_fcnn_classify';
    beforePadding = 0;

    trainList = importdata('trainList.data','\n',1000);
    testList = importdata('testList.data','\n',1000);

    trDataPath = strcat('results/all_img_' , savePrefix);
    trLabelPath = strcat('results/all_label_' , savePrefix);
    teDataPath = strcat('results/all_testimg_' , savePrefix);
    teLabelPath = strcat('results/all_testlabel_' , savePrefix);

    
    trainData = storeAllImage(trainList,beforePadding);
    save(strcat(trDataPath,'.mat'),'trainData','-v7.3');
    fprintf('successfully save:%s \n',trDataPath);
    
    testData = storeAllImage(testList,beforePadding);
    save(strcat(teDataPath,'.mat'),'testData','-v7.3');
    fprintf('successfully saved:%s \n',teDataPath);
    
    trLabel = storeAllSegmenttion(trainList,beforePadding);
    save(strcat(trLabelPath,'.mat'),'trLabel','-v7.3');
    fprintf('seg successfully saved:%s \n',trLabelPath);
    
    teLabel = storeAllSegmenttion(testList,beforePadding);
    save(strcat(teLabelPath,'.mat'),'teLabel','-v7.3');
    fprintf('seg successfully saved:%s \n',teLabelPath);
    
    
    
    function[scanData] = storeAllImage(list,padding)
        scanNumber = size(list,1);
        scanData = uint8(zeros(scanNumber*4, 1,(56+padding*2),512,512));
        for i=1:scanNumber
            scanId = list(i);
            scanPath = strcat(imageRootPath,'/',scanId(1),'/imgs/*.jpeg');
            images = dir(char(scanPath));
            fprintf('start load image:%s \n',char(scanId));
            scanNum = size(images,1);
            for j=1:(scanNum-3)
                imageId = images(j).name;
                imagePath = strcat(imageRootPath,'/',scanId(1),'/imgs/',imageId);
                scanImg = imread(char(imagePath));
                if ndims(scanImg)~=2
                    scanImg = scanImg(:,:,2);
                end
                scanData((i-1)*4+1, 1,(padding+j), :, :) = scanImg;
                
                flipScanImg = fliplr(scanImg);
                scanData((i-1)*4+2, 1, (padding+j), :,:) = flipScanImg;
                
                flipScanImg2 = flipud(scanImg);
                scanData((i-1)*4+3, 1, (padding+j), :,:) = flipScanImg2;
                
                flipScanImg3 = flipud(flipScanImg);
                scanData((i-1)*4+4, 1, (padding+j), :,:) = flipScanImg3;
                
                
            end
            
            
        end
        
    end

    function[scanData]= storeAllSegmenttion(list,padding)
        
        scanNumber = size(list,1);
        scanData = uint8(zeros(scanNumber*4,56));
        for i=1:scanNumber
            scanId = list(i);
            scanPath = strcat(annotationPath,'/',scanId(1),'/*.png');
            images = dir(char(scanPath));
            fprintf('start load segmentation:%s \n',char(scanId));
            scanNum = size(images,1);
            for j=1:scanNum
                imageId = images(j).name;
                imagePath = strcat(annotationPath,'/',scanId(1),'/',imageId);
                
                segMap = segBitmap(char(imagePath));
                
                flag = 0;
                if sum(segMap(:))>0
                    flag = 1;
                end
                
                
                currendId = str2double(imageId(end-6:end-4));
                scanData((i-1)*4+1,(padding+currendId)) = flag;
                
                
                scanData((i-1)*4+2, (padding+currendId), :,:) = flag;
                
                scanData((i-1)*4+3, (padding+currendId), :,:) = flag;
                
                scanData((i-1)*4+4, (padding+currendId), :,:) = flag;
                
                
            end
            
            
        end
        
        
    end

    function[annotation]= segBitmap(link)
        X = imread(link);
        annotation = X(:, :, 1);
        annotation = bitor(annotation, bitshift(X(:, :, 2), 8));
        annotation = bitor(annotation, bitshift(X(:, :, 3), 16));
    end



end

