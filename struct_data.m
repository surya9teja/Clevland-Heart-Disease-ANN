function data_struct = struct_data(file_name)
% Description : Helper function to arrange the data for MLP net
% Loading the file, to convert into categorical data 
data_struct = load(file_name);
setdemorandstream(672880951);

%Catgorizing the chest pain column
x2 = categorical(data_struct.x(:,3));
categories(x2);
x2_ = dummyvar(x2);
data_struct.x = [data_struct.x(:,1:2) x2_ data_struct.x(:,4:end)];

%Catgorizing the rest ECG column
x3 = categorical(data_struct.x(:,10));
categories(x3);
x3_ = dummyvar(x3);
data_struct.x = [data_struct.x(:,1:9) x3_ data_struct.x(:,11:end)]; 

% Catgorizing the slope column
x4 = categorical(data_struct.x(:,16));
categories(x4);
x4_ = dummyvar(x4);
data_struct.x = [data_struct.x(:,1:15) x4_ data_struct.x(:,17:end)];

% Catgorizing the Cholestrol column
x5 = categorical(data_struct.x(:,19));
categories(x5);
x5_ = dummyvar(x5);
data_struct.x = [data_struct.x(:,1:18) x5_ data_struct.x(:,20:end)];

% Catgorizing the thal column
x6 = categorical(data_struct.x(:,23));
categories(x6);
x6_ = dummyvar(x6);
data_struct.x = [data_struct.x(:,1:22) x6_ data_struct.x(:,24:end)];

%Catgorizing the target column
t1 = categorical(data_struct.t(:,1));
categories(t1);
data_struct.t = dummyvar(t1);

%merging input and ountput

full_data = [data_struct.x, data_struct.t];

% dividing the data for training, testing and validation
[trainInd,valInd,testInd] = dividerand(full_data',85,15,15);

% Rearraging the data for network input
data_struct.input_count = size(data_struct.x,2);
data_struct.output_count = size(data_struct.t,2);
data_struct.test.input = testInd(1:end-3,:);
data_struct.test.output = testInd(end-2:end,:);
data_struct.test_count = length(data_struct.test.input);
data_struct.training.input = trainInd(1:end-3,:);
data_struct.training.output = trainInd(end-2:end,:);
data_struct.training_count = length(data_struct.training.input);
data_struct.validation.input = valInd(1:end-3,:);
data_struct.validation.output = valInd(end-2:end,:);
data_struct.validation_count = length(data_struct.validation.input);

data_struct.training.bias_i = ones(data_struct.training_count, 1);
data_struct.validation.bias_i = ones(data_struct.validation_count, 1);
data_struct.test.bias_i = ones(data_struct.test_count, 1);
data_struct.training.bias_o = ones(data_struct.training_count, 1);
data_struct.validation.bias_o = ones(data_struct.validation_count, 1);
data_struct.test.bias_o = ones(data_struct.test_count, 1);
data_struct.training.count = data_struct.training_count;
data_struct.validation.count = data_struct.validation_count;
data_struct.test.count = data_struct.test_count;
% save('heart_disease_cat.mat', 'data_struct');
% save('heart_disease_cat.mat','data_struct.input_count','data_struct.output_count','data_struct.test.input'...
%     ,'data_struct.test.output','data_struct.test_count','data_struct.training.input','data_struct.training.output'...
%     ,'data_struct.training_count','data_struct.validation.input','data_struct.validation.output','data_struct.validation_count');
