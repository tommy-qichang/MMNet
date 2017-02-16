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
require 'mattorch'

local matDataPrefix = '';

local Provider = class('Provider');

function Provider:__init(trainSize, testSize)
   print '==> load dataset into trainData/testData'
--
--
--    self.allCropImagesPositive = mattorch.load('matlab/'.. matDataPrefix ..'bioimg_data_positive.mat');
--    self.allCropImagesNegative = mattorch.load('matlab/'.. matDataPrefix ..'bioimg_data_negative.mat');
----    local allCropImagesWeight = matio.load('matlab/'..matDataPrefix..'bioimg_weight.mat');
--    print '==> finish load train data, start load test data...';
--    self.testImagesPositive = mattorch.load('matlab/'..matDataPrefix..'bioimg_test_data_positive.mat');
--    self.testImagesNegative = mattorch.load('matlab/'..matDataPrefix..'bioimg_test_data_negative.mat');
----    self.testImagesWeight = matio.load('matlab/'..matDataPrefix..'bioimg_test_weight.mat');
--
--    self.allCropImagesPositive.trDataPositive = self.allCropImagesPositive.trDataPositive:t();
--    self.allCropImagesNegative.trDataNegative = self.allCropImagesNegative.trDataNegative:t();
--
--    self.testImagesPositive.teDataPositive = self.testImagesPositive.teDataPositive:t();
--    self.testImagesNegative.teDataNegative = self.testImagesNegative.teDataNegative:t();
--    print '==> finish load dataset, start clean data...'


--    allCropImages.trData[{{mask},{}}]

end

function file_exists(name)
    local f=io.open(name,"r")
    if f~=nil then io.close(f) return true else return false end
end

local datasetName = 'provider_voxel';
local datasetPath = 'preprocessing/results/'..datasetName..'.t7';

function Provider:load()
    print '==> load dataset into trainData/testData';

    if file_exists(datasetPath)then
        print '==> file already exist, directly load train data';

        self.dataset = torch.load(datasetPath);
        return true;
    else
        self.dataset = {};
    end

    local stamp = '_1108';
    for i=1,2 do
        local trNegDataPath = 'preprocessing/results/patch_train_neg_'..i..stamp..'.mat';
        local trPosDataPath = 'preprocessing/results/patch_train_pos_'..i..stamp..'.mat';

        print ('==> begin load trPosData idx:'..i..'');
        local partTrNegData = mattorch.load(trNegDataPath);
        local partTrPosData = mattorch.load(trPosDataPath);

        print ('==> finish load train data, combine it...');
        if self.dataset.allTrNegData then
            self.dataset.allTrNegData= torch.cat(self.dataset.allTrNegData,partTrNegData.allCropImagesNeg,4);
            self.dataset.allTrPosData= torch.cat(self.dataset.allTrPosData,partTrPosData.allCropImagesPos,4);
        else
            self.dataset.allTrNegData = partTrNegData.allCropImagesNeg;
            self.dataset.allTrPosData = partTrPosData.allCropImagesPos;
        end
    end

    self.dataset.allTrNegData = self.dataset.allTrNegData:permute(4,3,2,1);
    self.dataset.allTrPosData = self.dataset.allTrPosData:permute(4,3,2,1);

    print(self.dataset.allTrNegData:size());
    print(self.dataset.allTrPosData:size());


    for i=1,3 do
        local teNegDataPath = 'preprocessing/results/patch_test_neg_'..i..stamp..'.mat';
        local tePosDataPath = 'preprocessing/results/patch_test_pos_'..i..stamp..'.mat';

        print ('==> begin load tePosData idx:'..i..'');
        local partTeNegData = mattorch.load(teNegDataPath);
        local partTePosData = nil;
        if i~=3 then
            partTePosData = mattorch.load(tePosDataPath);
        end

        print ('==> finish load test data, combine it...');
        if self.dataset.allTeNegData then
            self.dataset.allTeNegData= torch.cat(self.dataset.allTeNegData,partTeNegData.allCropImagesNeg,4);

            if partTePosData ~= nil then
                self.dataset.allTePosData= torch.cat(self.dataset.allTePosData,partTePosData.allCropImagesPos,4);
            end

        else
            self.dataset.allTePosData = partTePosData.allCropImagesPos;
            self.dataset.allTeNegData = partTeNegData.allCropImagesNeg;
        end
    end

    self.dataset.allTeNegData = self.dataset.allTeNegData:permute(4,3,2,1);
    self.dataset.allTePosData = self.dataset.allTePosData:permute(4,3,2,1);

    print(self.dataset.allTeNegData:size());
    print(self.dataset.allTePosData:size());


end



function Provider:update()

    local origPositiveTrainSize = self.dataset.allTrPosData:size(1);
    local origTrainSize = origPositiveTrainSize*2;

    local origPositiveTestSize = self.dataset.allTePosData:size(1);
    local origNegativeTestSize = self.dataset.allTeNegData:size(1);
    local origTestSize = origPositiveTestSize + origNegativeTestSize;

    local zLength = self.dataset.allTrPosData:size(2);
    local yLength = self.dataset.allTrPosData:size(3);
    local xLength = self.dataset.allTrPosData:size(4);


    print(('==> origTrainSize: %d; origTestSize: %d; z:%d; y:%d; x:%d \n'):format(origTrainSize, origTestSize, zLength, yLength, xLength));

    local indices = torch.randperm(self.dataset.allTrNegData:size(1))[{{1,origPositiveTrainSize}}];

    local trainNegativeData = self.dataset.allTrNegData:index(1,indices:long()):clone();
    local trainPositiveData = self.dataset.allTrPosData:clone();

    print('==> finish training data resize...');

    local testNegativeData = self.dataset.allTeNegData:clone();
    local testPositiveData = self.dataset.allTePosData:clone();

    print('==> finish testing data resize...');
    local trainSize = trainSize or origTrainSize;
    local testSize = testSize or origTestSize;

    self.dataset.trainData = {
        data = torch.Tensor(origTrainSize, zLength, yLength, xLength),
        labels = torch.Tensor(origTrainSize),
        size = function() return trainSize end
    }
    self.dataset.testData = {
        data = torch.Tensor(origTestSize, zLength, yLength, xLength),
        labels = torch.Tensor(origTestSize),
        size = function() return testSize end
    }

    --build training data, first part is part of negative data from negative data set;second part is all positive data.
    print('===> combine train data...');

    local trainData = self.dataset.trainData;
    trainData.data[{ { 1, origPositiveTrainSize } }] = trainNegativeData:float();
    trainData.data[{ { origPositiveTrainSize+1, origTrainSize } }] = trainPositiveData:float();

    trainData.labels[{ { 1, origPositiveTrainSize } }] = torch.Tensor(origPositiveTrainSize):fill(0):float();
    trainData.labels[{ { origPositiveTrainSize+1, origTrainSize  } }] = torch.Tensor(origPositiveTrainSize):fill(1):float();

    --build test data.
    print('===> combine test data...')
    local testData = self.dataset.testData;
    testData.data[{{1,origNegativeTestSize}}] = testNegativeData:float();
    testData.data[{{origNegativeTestSize+1, origTestSize}}] = testPositiveData:float();

    print('===> finish data combine');

    testData.labels[{ { 1, origNegativeTestSize } }] = torch.Tensor(origNegativeTestSize):fill(0):float();
    testData.labels[{ { origNegativeTestSize+1, origTestSize  } }] = torch.Tensor(origPositiveTestSize):fill(1):float();

    print('==> finish all combine');

    trainData.labels = trainData.labels + 1;
    testData.labels = testData.labels + 1;


    print '==> train data and test data size:::';
    print (trainData.data:size());
    print (testData.data:size());
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
provider:load();
provider:update();
provider:normalize();
print '==> save provider.t7'
torch.save(datasetPath, provider);


