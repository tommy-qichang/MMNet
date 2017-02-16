--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 10:05 AM
--
require 'nn';
require 'image';
require 'xlua';
require 'math';
local class = require 'class'
matio = require 'matio'


local Provider = class('Provider');

function Provider:__init(trainSize, testSize)
    print '==> load dataset into trainData/testData'

    for i=0,10 do
        local trDataPath = 'preprocessing'
    end


    local allCropImages = matio.load('matlab/bioimg_data.mat');
    local allCropImagesWeight = matio.load('matlab/bioimg_weight.mat');
    print '==> finish load train data, start load test data...';
    local testImages = matio.load('matlab/bioimg_test_data.mat');
    local testImagesWeight = matio.load('matlab/bioimg_test_weight.mat');

    print '==> finish load dataset, start clean data...'

    local origTrainSize = allCropImages.trData:size(1);
    local origTestSize = testImages.teData:size(1);
    local dataLength = allCropImages.trData:size(2);
    local xLength = math.sqrt(dataLength);


    local trainSize = trainSize or origTrainSize;
    local testSize = testSize or origTestSize;

    print(('==> origTrainSize: %d; origTestSize: %d; dataLength:%d; xlength:%d'):format(origTrainSize, origTestSize, dataLength, xLength));

    self.trainData = {
        data = torch.Tensor(origTrainSize, dataLength),
        labels = torch.Tensor(origTrainSize),
        size = function() return trainSize end
    }
    self.testData = {
        data = torch.Tensor(origTestSize, dataLength),
        labels = torch.Tensor(origTestSize),
        size = function() return testSize end
    }

    local trainData = self.trainData;
    trainData.data[{ { 1, origTrainSize } }] = allCropImages.trData:float();
    trainData.labels[{ { 1, origTrainSize } }] = allCropImagesWeight.trWeight:select(2, 5):float();

    local testData = self.testData;
    testData.data[{ { 1, origTestSize } }] = testImages.teData:float();
    testData.labels[{ { 1, origTestSize } }] = testImagesWeight.teWeight:select(2, 5):float();

    --resize if using small dataset.
    trainData.data = trainData.data[{ { 1, trainSize } }];
    trainData.labels = trainData.labels[{ { 1, trainSize } }];

    testData.data = testData.data[{ { 1, testSize } }];
    testData.labels = testData.labels[{ { 1, testSize } }];

    trainData.labels = trainData.labels + 1;
    testData.labels = testData.labels + 1;

    -- reshape data & transpose
    trainData.data = trainData.data:reshape(trainSize, 1, xLength, xLength):transpose(3, 4);
    testData.data = testData.data:reshape(testSize, 1, xLength, xLength):transpose(3, 4);
end

function Provider:normalize()
    local trainData = self.trainData;
    local testData = self.testData;

    collectgarbage();
    print '==> pre-processing data'

    local mean = trainData.data:select(2, 1):mean();
    local std = trainData.data:select(2, 1):std();
    trainData.data:select(2, 1):add(-mean);
    trainData.data:select(2, 1):div(std);
    trainData.mean = mean;
    trainData.std = std;

    testData.data:select(2, 1):add(-mean);
    testData.data:select(2, 1):div(std);
    testData.mean = mean;
    testData.std = std;
end

provider = Provider();
provider:normalize();
print '==> save provider.t7'
torch.save('provider.t7', provider);

