function [T] = augmentation(file_path, Fs, t_rms_bounds, f_d_bounds, k_factor_bounds, N)
    rng(1234);

    label = h5read(file_path, '/label');
    data = h5read(file_path, '/data');
    rssi = h5read(file_path, '/rssi');

    data = repmat(data, 1, N);
    label = repmat(label, 1, N);
    rssi = repmat(rssi, 1, N);

    data_aug = zeros(size(data, 1)/2, length(label));

    for frame_idx = 1:length(label)
        fprintf('.');
        if rem(frame_idx - 1, 100) == 0
            fprintf('\n')
            fprintf(num2str(frame_idx - 1))
        end

        frame_iq = squeeze(data(:, frame_idx));
        X = frame_iq(1:2:end) + 1i * frame_iq(2:2:end);

        [X_aug, ~] = augmentation_run(X, Fs, t_rms_bounds, f_d_bounds, k_factor_bounds);

        data_aug(:, frame_idx) = X_aug;
    end
    fprintf('\n');

    T = struct();
    T.('data_aug') = data_aug;
    T.('label_aug') = label;
    T.('rssi_aug') = rssi;
end

function [sig_out, myPathGain] = augmentation_run(sig_in, Fs, t_rms_bounds, f_d_bounds, k_factor_bounds)
    Ts = 1 / Fs;
    t_rms = get_random_value(t_rms_bounds(1), t_rms_bounds(2), 1e-9);
    f_d = get_random_value(f_d_bounds(1), f_d_bounds(2), 1);
    k_factor = get_random_value(k_factor_bounds(1), k_factor_bounds(2), 1);

    [avg_path_gains, path_delays]= get_exp_pdp(t_rms, Ts);
    
    wirelessChan = comm.RicianChannel(...
        'SampleRate', Fs,...
        'KFactor', k_factor,...
        'MaximumDopplerShift', f_d,...
        'PathDelays',path_delays,...
        'AveragePathGains', avg_path_gains,...
        'DopplerSpectrum', doppler('Jakes'),...
        'PathGainsOutputPort', true);
    
    delay = info(wirelessChan).ChannelFilterDelay;
    
    chInput = [sig_in;zeros(50,1)];
    [chOut, myPathGain] = wirelessChan(chInput);
    sig_out = chOut(delay + 1 : delay + length(sig_in));
end

function [avg_path_gains, path_delays] = get_exp_pdp(t_rms, Ts)
    A_dB = -30;
    sigma_tau = t_rms; 
    A=10^(A_dB/10);
    lmax=ceil(-t_rms*log(A)/Ts); % Eq.(2.2)

    % Exponential PDP
    p=0:lmax; 
    path_delays = p*Ts;

    p = (1/sigma_tau)*exp(-p*Ts/sigma_tau);
    p_norm = p/sum(p);

    avg_path_gains = 10*log10(p_norm); % convert to dB
end

function [random_value] = get_random_value(val_min, val_max, multiplier)
    random_value = ((val_max-val_min).*rand(1) + val_min) * multiplier;
end