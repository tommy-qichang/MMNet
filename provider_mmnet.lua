--
-- Created by IntelliJ IDEA.
-- User: changqi
-- Date: 3/14/16
-- Time: 10:05 AM
--
require 'nn';
require 'xlua';
require 'math';
local class = require 'class'
require 'mattorch'

local matDataPrefix = '';

local Provider = class('Provider');
local datasetName = nil;
local datasetPath = nil;
local id = 1;

function Provider:__init(_id)
    id = _id
    datasetName = 'provider_mmnet_';
    datasetPath = 'training/'..datasetName.._id..'.t7';
    print '==> load dataset into trainData/testData'
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

--local datasetName = 'provider_mmnet';
--local datasetPath = 'training/'..datasetName..'.t7';
--local needNorm = true;

function Provider:load()
    print '###> load dataset into trainData/testData';

    if file_exists(datasetPath)then
        print '==> file already exist, directly load train data';

        self.dataset = torch.load(datasetPath).dataset;
        return false;
    else
        self.dataset = {};
    end

    local stamp = '0114_mmnet_'..id;
    local trDataPath = 'preprocessing/clean_data/all_img_'..stamp..'.mat';
    local trLabelPath = 'preprocessing/clean_data/all_label_'..stamp..'.mat';

    print ('==> begin load trPosData ');
    local trData = mattorch.load(trDataPath);
    local trLabel = mattorch.load(trLabelPath);

    self.dataset.trainData = {
    }
    self.dataset.testData = {
    }


    print ('==> finish load train data, combine it...');

    self.dataset.trainData.data = trData.trainData:permute(5,4,3,2,1):float();
    self.dataset.trainData.label = trLabel.trLabel:permute(5,4,3,2,1):float();


    local teDataPath = 'preprocessing/results/all_testimg_'..stamp..'.mat';
    local teLabelPath = 'preprocessing/results/all_testlabel_'..stamp..'.mat';

    print ('==> begin load tePosData ');
    local teData = mattorch.load(teDataPath);
    local teLabel = mattorch.load(teLabelPath);

    print ('==> finish load test data, combine it...');
    self.dataset.testData.data = teData.testData:permute(5,4,3,2,1):float();
    self.dataset.testData.label = teLabel.teLabel:permute(5,4,3,2,1):float();


    self.dataset.trainData.label = self.dataset.trainData.label + 1;
    self.dataset.testData.label = self.dataset.testData.label + 1;

    return true;
end




function Provider:normalize()
    local trainData = self.dataset.trainData;
    local testData = self.dataset.testData;

    collectgarbage();
    print '###> pre-processing data'

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
saved = provider:load();
if needNorm == true then
    provider:normalize();
end

if saved then
    print '==> save provider.t7'
    torch.save(datasetPath, provider);
else
    print '==> already saved do nothing.'
end



