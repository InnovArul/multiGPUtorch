require 'nn'
require 'cutorch'
require 'cunn'
require 'image'
require 'torch'
require 'nngraph'
dofile 'util.lua'

--
-- Single lengthy model 
--
function createSingleModel()
  -- hidden units, filter sizes (for ConvNet only):
  nfeats = 3
  nstates = {64,64,128}
  filtsize = 5
  poolsize = 2

  -- 2-class problem
  noutputs = 2

  model = nn.Sequential()

  -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
  model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
  model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
  model:add(nn.ReLU())
  model:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

  -- stage 3 : standard 2-layer neural network
  model:add(nn.View(nstates[2]*filtsize*filtsize))
  model:add(nn.Dropout(0.5))
  model:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
  model:add(nn.ReLU())
  model:add(nn.Linear(nstates[3], noutputs))
  model:add(nn.LogSoftMax())
  
  return model:cuda()

end


--
-- Data preparation for Single network
--
function prepareDataForSingleNetwork()
  count = 10000
  input = torch.Tensor(count, 3, 32, 32):rand(10000, 3, 32, 32):cuda()
  target = torch.Tensor(count):fill(1):cuda()
  return input, target
end

--
-- Siamese model creation
--
function createSiameseModel()
  -- hidden units, filter sizes (for ConvNet only):
  nfeats = 3
  nstates = {64,64,128}
  filtsize = 5
  poolsize = 2

  -- 2-class problem
  noutputs = 2

  --PIPELINE 1
  -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
  pipe1_conv1 = nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize)()
  pipe1_relu1 = nn.ReLU()(pipe1_conv1)
  pipe1_pool1 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(pipe1_relu1)

  -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
  pipe1_conv2 = nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize)(pipe1_pool1)
  pipe1_relu2 = nn.ReLU()(pipe1_conv2)
  pipe1_pool2 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(pipe1_relu2)
  
  --PIPELINE 2
  -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
  pipe2_conv1 = nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize)()
  pipe2_relu1 = nn.ReLU()(pipe2_conv1)
  pipe2_pool1 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(pipe2_relu1)

  -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
  pipe2_conv2 = nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize)(pipe2_pool1)
  pipe2_relu2 = nn.ReLU()(pipe2_conv2)
  pipe2_pool2 = nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize)(pipe2_relu2)  
 
  join = nn.JoinTable(1)({pipe1_pool2, pipe2_pool2})
  
  -- stage 3 : standard 2-layer neural network
  view = nn.View(2 * nstates[2]*filtsize*filtsize)(join)
  dropout = nn.Dropout(0.5)(view)
  lin1 = nn.Linear(2 * nstates[2]*filtsize*filtsize, nstates[3])(dropout)
  relu = nn.ReLU()(lin1)
  lin2 = nn.Linear(nstates[3], noutputs)(relu)
  softmax = nn.LogSoftMax()(lin2)
  
  model = nn.gModule({pipe1_conv1, pipe2_conv1}, {softmax})
  graph.dot(model.fg, 'modelSiamese',  'modelSiamese')
  
  return model:cuda()

end


--
-- Data preparation for Siamese network
--
function prepareDataForSiameseNetwork()
  count = 10000
  input = {}
  
  for index = 1, count do
    table.insert(input, {torch.Tensor(3, 32, 32):rand(3, 32, 32):cuda(),
                        torch.Tensor(3, 32, 32):rand(3, 32, 32):cuda()}
                      )
  end
  
  target = torch.Tensor(count):fill(1):cuda()
  return input, target
end
