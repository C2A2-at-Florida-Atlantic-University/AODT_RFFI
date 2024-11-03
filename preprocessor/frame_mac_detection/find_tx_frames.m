% This source code is a slightly refactored and simplified
% version of the 802.11 waveform recovery & analysis demo:
% Ref: https://www.mathworks.com/help/wlan/ug/recover-and-analyze-packets-in-802-11-waveform.html

function T = find_tx_frames(filepath, bw, samp_rate, search_mac_tx, preamble_len, enable_equalization)
    nht = wlanNonHTConfig;

    if enable_equalization == true
        fprintf('Equalization enabled.\n');
    else 
        frintf('Equalization disabled.\n')
    end

    X = read_iq(filepath);
    % figure;
    % plot(1:length(X), real(X));
    % X = X(1:floor(length(X))); % TODO: only for larger files

    % 1. Analyze the waveform
    analyzer = WaveformAnalyzer();
    process(analyzer, X, bw, samp_rate);
    
    % 2. Extract MAC info for a given frame
    preamble_bounds = {};
    preamble_iq = {};
    rssi = {};
    macs = {};
    
    fprintf('Searching for %s\n', search_mac_tx);

    for mac_i = 1:size(analyzer.Results, 2)
        item = analyzer.Results{mac_i};
        if item.MAC.Processed == 0
            continue;
        end

        mac_summary = macSummary(analyzer, mac_i, false);
        if isempty(mac_summary)
            continue;
        end
    
        frame_tx_mac = parse_mac_address(mac_summary{1, 2}); % MAC address of the emitter

        if strcmp(frame_tx_mac, 'NA') | strcmp(frame_tx_mac, 'Unknown') | length(frame_tx_mac) ~= 17
            continue;
        end
    
        % Filter frames based on the TX MAC address
        if length(search_mac_tx) == 0 | strcmp(frame_tx_mac, search_mac_tx)
            % Extract start & end indexes for the frame preamble
            x = analyzer.Results{mac_i};
    
            samples_start = x.PacketOffset;
        
            % Load full frame if preamble length isn't specified
            if preamble_len == -1
                samples_end = samples_start + x.NumRxSamples;
            else
                samples_end = samples_start + preamble_len - 1;
            end

            X_preamble = X(samples_start:samples_end);

            if enable_equalization == true
                fprintf('!');
                % Resample from 25e6 to 20e6 Msps
                X_preamble = resample(X_preamble, 4, 5);
                % Equalize the preamble and reapply CFO
                X_preamble = equalize(X_preamble, nht);
                % Resample from 20e6 to 25e6
                X_preamble = resample(X_preamble, 5, 4);
            else
                fprintf('.');
            end

            preamble_iq{end+1} = X_preamble;
            preamble_bounds{end+1} = [samples_start, samples_end];
            rssi{end+1} = round(10*log10(x.PacketPower),2);
            macs(end+1) = {frame_tx_mac};
        end
    end
    fprintf('\n');

    fprintf('Found %i TX frames.\n', length(preamble_bounds));

    T = struct();
    T.('preamble_bounds') = preamble_bounds;
    T.('preamble_iq') = preamble_iq;
    T.('rssi') = rssi;
    T.('macs') = macs;
end

function [formattedMac] = parse_mac_address(mac)
    macAddress = sprintf('%s', mac);

    % Validate the input
    if strlength(macAddress) ~= 12
        formattedMac = macAddress;
        return;
    end
    
    % Convert to lowercase
    macAddress = lower(macAddress);
    
    % Insert colons to separate octets
    formattedMac = sprintf('%s:%s:%s:%s:%s:%s', ...
        macAddress(1:2), macAddress(3:4), macAddress(5:6), ...
        macAddress(7:8), macAddress(9:10), macAddress(11:12));
end

function [X] = read_iq(filename, count)
    m = nargchk (1,2,nargin);
    if (m)
        usage(m);
    end
    
    if (nargin < 2)
        count = Inf;
    end
    
    f = fopen (filename, 'rb');
    if (f < 0)
        X = 0;
    else
        t = fread(f, [2, count], 'float');
        fclose (f);
        X = t(1,:) + t(2,:) * i;
        [r, c] = size(X);
        X = reshape(X, c, r);
    end
end

% Estimate CSI and equalize the frame
function [pkt_stf_ltf_fo] = equalize(pkt, nht)
    % Determine indexes of STF and LTF sequences
    stf_ind = wlanFieldIndices(nht,'L-STF');
    ltf_ind = wlanFieldIndices(nht,'L-LTF');
    pkt_stf = pkt(stf_ind(1):stf_ind(2));

    freqOffsetEst1 = wlanCoarseCFOEstimate(pkt_stf,'CBW20');
    pkt = pkt.*exp(1j*(1:length(pkt))'/20e6*2*pi*-freqOffsetEst1 );
    pkt_ltf = pkt(ltf_ind(1):ltf_ind(2));

    freqOffsetEst2 = wlanFineCFOEstimate(pkt_ltf,'CBW20');
    pkt = pkt.*exp(1j*(1:length(pkt))'/20e6*2*pi*-freqOffsetEst2);

    % Specify indexes of null & pilot subcarriers
    datIndx= [39:64 2:27];

    % Get IQ samples of the STF sequence
    pkt_stf = pkt(stf_ind(1):stf_ind(2));

    % Get IQ samples of the LTF sequence
    pkt_ltf = pkt(ltf_ind(1):ltf_ind(2));

    % Demodulate LTF, estimate noise, CSI
    demodSig = wlanLLTFDemodulate(pkt_ltf,nht);
    nVar = helperNoiseEstimate(demodSig, nht.ChannelBandwidth, 1);
    
    % Merge the preamble, reshape into a (64x5) array, run FFT
    pkt_stf_ltf = [pkt_stf; pkt_ltf];
    pkt_stf_ltf_resh = reshape(pkt_stf_ltf,64,5);
    pkt_stf_ltf_freq = fft(pkt_stf_ltf_resh,64);
    pkt_stf_ltf_freq = pkt_stf_ltf_freq(datIndx,:);

    % Estimate CSI, perform equalization of the preamble
    h1 = wlanLLTFChannelEstimate(demodSig,nht);
    pkt_stf_ltf_freq_eq = pkt_stf_ltf_freq.*conj(h1)./(conj(h1).*h1+nVar);

    pkt_stf_ltf_freq_eq_all = zeros(64,5);
    pkt_stf_ltf_freq_eq_all(datIndx,:) = pkt_stf_ltf_freq_eq;
    pkt_stf_ltf_eq = ifft(pkt_stf_ltf_freq_eq_all,64);
    pkt_stf_ltf_eq = pkt_stf_ltf_eq(:);
    
    pkt_stf_ltf_fo = pkt_stf_ltf_eq .*exp(1j*(1:length(pkt_stf_ltf_eq))'/20e6*2*pi*(freqOffsetEst1+freqOffsetEst2));
end