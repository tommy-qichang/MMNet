function storeAll_MMNet(id)
    imageRootPath = 'images';
    annotationPath = 'annotations';
    savePrefix=strcat('0114_mmnet_',id);
    beforePadding = 0;

    trainList = importdata(strcat('trainList-',id,'.data'),'\n',1000);
    testList = importdata(strcat('testList-',id,'.data'),'\n',1000);

    trDataPath = strcat('clean_data/all_img_' , savePrefix);
    trLabelPath = strcat('clean_data/all_label_' , savePrefix);
    teDataPath = strcat('clean_data/all_testimg_' , savePrefix);
    teLabelPath = strcat('clean_data/all_testlabel_' , savePrefix);

    
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
        scanData = uint8(zeros(scanNumber,1,(40+padding*2),200,200));
        for i=1:scanNumber
            scanId = list(i);
            scanPath = strcat(imageRootPath,'/',scanId(1),'/imgs/*.jpeg');
            images = dir(char(scanPath));
            fprintf('start load image:%s \n',char(scanId));
            scanNum = size(images,1);
            for j=1:(scanNum)
                imageId = images(j).name;
                imagePath = strcat(imageRootPath,'/',scanId(1),'/imgs/',imageId);
                scanImg = imread(char(imagePath));
                if ndims(scanImg)~=2
                    scanImg = scanImg(:,:,2);
                end
                scanImg = imresize(scanImg,[200 200]);
                scanData(i,:,(padding+j),:,:) = scanImg;
                
            end
            
            
        end
        

    end

    function[scanData]= storeAllSegmenttion(list,padding)
        
        scanNumber = size(list,1);
        scanData = uint8(zeros(scanNumber,(40+padding*2),200,200));
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
                
                segMap = imresize(segMap,[200 200],'nearest');
                
                currendId = str2double(imageId(end-6:end-4));
                scanData(i,(padding+currendId),:,:) = segMap;
                
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

