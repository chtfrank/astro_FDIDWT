clc;
close all;
clear;


new_col = [2,0,5,6,10,4,7,11,3,8,1,9,12]+1;
n_core = 2;
mission_str_list = {'Mission_A', 'Mission_B'};



for mission_indx = 1:2
    input_base_dir = ['RESULTS/', mission_str_list{mission_indx}, '/FDIDWT'];
    for repeat = 1:10
        train_path = [input_base_dir, '/split_', num2str(repeat), '_trainX.txt'];
        validation_path = [input_base_dir, '/split_', num2str(repeat), '_validationX.txt'];
        test_path = [input_base_dir, '/split_', num2str(repeat), '_testX.txt'];
        predict_path = [input_base_dir, '/split_', num2str(repeat), '_predictX.txt'];
        save_dir_str = [input_base_dir, '/step2_4_2_iwt_outputs/split_', num2str(repeat)];


        train_samples = textread(train_path);
        validation_samples = textread(validation_path);
        test_samples = textread(test_path);
        predict_samples = textread(predict_path);

        train_attr = train_samples(:, 1:13);
        validation_attr = validation_samples(:, 1:13);
        test_attr = test_samples(:, 1:13);
        predict_attr = predict_samples(:, 1:13);

        new_train_attr = train_attr(:, new_col);
        new_validation_attr = validation_attr(:, new_col);
        new_test_attr = test_attr(:, new_col);
        new_predict_attr = predict_attr(:, new_col);

        [n_train, n_attr] = size(new_train_attr);
        n_validation = size(new_validation_attr, 1);
        n_test = size(new_test_attr, 1);
        n_predict = size(new_predict_attr, 1);




        % --------------- level 1 --------------- 
        L = [7,7,13; 7,7,12; 7,7,11; 7,7,10; 7,7,9; 7,7,8; 7,7,7; 7,7,6; 7,7,5; 7,7,4; 7,7,3; 7,7,2; 7,7,1;];
        wavelets = {'db1', 'db2', 'db2', 'db3', 'db3', 'db4', 'db4', 'db5', 'db5', 'db6', 'db6', 'db7', 'db7'};
        level = 'level1';
        n_pad = 1;

        for i = 1:size(L,1)
            disp(['processing case ', num2str(i), '/ ', num2str(size(L,1))]);
            l = L(i, :);
            inL = L(i, end);
            wname = wavelets{i};
            
            train_data = zeros(n_train, inL);
            for j = 1:n_train
                c = [new_train_attr(j, 1:n_core), zeros(1, n_pad), new_train_attr(j, n_core+1:end)];
                train_data(j, :) = waverec(c, l, wname);
            end
            
            validation_data = zeros(n_validation, inL);
            for j = 1:n_validation
                c = [new_validation_attr(j, 1:n_core), zeros(1, n_pad), new_validation_attr(j, n_core+1:end)];
                validation_data(j, :) = waverec(c, l, wname);
            end
            
            test_data = zeros(n_test, inL);
            for j = 1:n_test
                c = [new_test_attr(j, 1:n_core), zeros(1, n_pad), new_test_attr(j, n_core+1:end)];
                test_data(j, :) = waverec(c, l, wname);
            end
            
            predict_data = zeros(n_predict, inL);
            for j = 1:n_predict
                c = [new_predict_attr(j, 1:n_core), zeros(1, n_pad), new_predict_attr(j, n_core+1:end)];
                predict_data(j, :) = waverec(c, l, wname);
            end
            
            % save data
            disp([level, ' saving outputDim = ', num2str(inL), ', wname = ', wname]);
            path_name = [save_dir_str, '/',level];
            if exist(path_name,'dir')==0
                folder = mkdir(path_name);
            end
            train_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_train.txt'], 'w');
            validation_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_validation.txt'], 'w');
            test_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_test.txt'], 'w');
            predict_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_un.txt'], 'w');
            for m = 1:n_train
                for n =  1:inL
                    fprintf(train_fid, '%f ', train_data(m, n));
                end
                fprintf(train_fid, '\n');
            end
            for m = 1:n_validation
                for n =  1:inL
                    fprintf(validation_fid, '%f ', validation_data(m, n));
                end
                fprintf(validation_fid, '\n');
            end
            for m = 1:n_test
                for n =  1:inL
                    fprintf(test_fid, '%f ', test_data(m, n));
                end
                fprintf(test_fid, '\n');
            end
            for m = 1:n_predict
                for n =  1:inL
                    fprintf(predict_fid, '%f ', predict_data(m, n));
                end
                fprintf(predict_fid, '\n');
            end
            fclose(train_fid);
            fclose(validation_fid);
            fclose(test_fid);
            fclose(predict_fid);
        end
        disp([level, ' done']);





        % --------------- level 3 --------------- 
        L = [3,3,3,4,5; 3,3,3,4,6;];
        wavelets = {'db2', 'db2'};
        level = 'level3';

        for i = 1:size(L,1)
            disp(['processing case ', num2str(i), '/ ', num2str(size(L,1))]);
            l = L(i, :);
            inL = L(i, end);
            wname = wavelets{i};
            
            train_data = zeros(n_train, inL);
            for j = 1:n_train
                c = new_train_attr(j, :);
                train_data(j, :) = waverec(c, l, wname);
            end
            
            validation_data = zeros(n_validation, inL);
            for j = 1:n_validation
                c = new_validation_attr(j, :);
                validation_data(j, :) = waverec(c, l, wname);
            end
            
            test_data = zeros(n_test, inL);
            for j = 1:n_test
                c = new_test_attr(j, :);
                test_data(j, :) = waverec(c, l, wname);
            end
            
            predict_data = zeros(n_predict, inL);
            for j = 1:n_predict
                c = new_predict_attr(j, :);
                predict_data(j, :) = waverec(c, l, wname);
            end
            
            % save data
            disp([level, ' saving outputDim = ', num2str(inL), ', wname = ', wname]);
            path_name = [save_dir_str, '/',level];
            if exist(path_name,'dir')==0
                folder = mkdir(path_name);
            end
            train_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_train.txt'], 'w');
            validation_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_validation.txt'], 'w');
            test_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_test.txt'], 'w');
            predict_fid = fopen([path_name,'/outputDim', num2str(inL), '_', wname, '_un.txt'], 'w');
            for m = 1:n_train
                for n =  1:inL
                    fprintf(train_fid, '%f ', train_data(m, n));
                end
                fprintf(train_fid, '\n');
            end
            for m = 1:n_validation
                for n =  1:inL
                    fprintf(validation_fid, '%f ', validation_data(m, n));
                end
                fprintf(validation_fid, '\n');
            end
            for m = 1:n_test
                for n =  1:inL
                    fprintf(test_fid, '%f ', test_data(m, n));
                end
                fprintf(test_fid, '\n');
            end
            for m = 1:n_predict
                for n =  1:inL
                    fprintf(predict_fid, '%f ', predict_data(m, n));
                end
                fprintf(predict_fid, '\n');
            end
            fclose(train_fid);
            fclose(validation_fid);
            fclose(test_fid);
            fclose(predict_fid);
        end
        disp([level, ' done']);

    end
end


