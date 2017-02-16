function transposeImg()
id = '0205103933';

imageRootPath = 'images';
annotationPath = 'annotations';
scanPath = strcat(imageRootPath,'/',id,'/imgs/*.jpeg');
images = dir(char(scanPath));
scanNum = size(images,1);
for j=1:(scanNum-3)
    imageId = images(j).name;
    imagePath = strcat(imageRootPath,'/',id,'/imgs/',imageId);
    scanImg = imread(char(imagePath));
    if ndims(scanImg)~=2
        scanImg = scanImg(:,:,2);
    end
    
    tScanImg = transpose(scanImg);
     imwrite(tScanImg,imagePath);
end

annoscanPath = strcat(annotationPath,'/',id,'/*.png');
images = dir(char(annoscanPath));
scanNum = size(images,1);
for j=1:scanNum
    imageId = images(j).name;
    imagePath = strcat(annotationPath,'/',id,'/',imageId);

    segMap = segBitmap(char(imagePath));
    segMap = transpose(segMap);
    
    revertBitmap(segMap, imagePath)

end


function[annotation]= segBitmap(link)
    X = imread(link);
    annotation = X(:, :, 1);
    annotation = bitor(annotation, bitshift(X(:, :, 2), 8));
    annotation = bitor(annotation, bitshift(X(:, :, 3), 16));
end

function revertBitmap(bitmap,link)
    X = cat(3,bitand(bitmap,255),...
        bitand(bitshift(bitmap,-8),255),...
        bitand(bitshift(bitmap,-16),255));
       imwrite(uint8(X),link);

end

end
