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

    trLabel = storeAllSegmenttion(trainList,beforePadding);
    
    teLabel = storeAllSegmenttion(testList,beforePadding);
    
    
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
                
                segMap(segMap>0)=1;
                if(sum(segMap(:))>=512*512/3)clear ; close all; clc
                    fprintf('The scan Number:%d, slide Id:%s \n',i,char(imageId));
                end
                
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

