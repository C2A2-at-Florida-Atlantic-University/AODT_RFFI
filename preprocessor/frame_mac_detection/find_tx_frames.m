% This source code is a slightly refactored and simplified
% version of the 802.11 waveform recovery & analysis demo:
% Ref: https://www.mathworks.com/help/wlan/ug/recover-and-analyze-packets-in-802-11-waveform.html

function T = find_tx_frames(filepath, bw, samp_rate, search_mac_tx, preamble_len)
    X = read_iq(filepath);
    X = X(1:floor(length(X) / 10)); % TODO: only for larger files

    % 1. Analyze the waveform
    analyzer = WaveformAnalyzer();
    process(analyzer, X, bw, samp_rate);
    
    % 2. Extract MAC info for a given frame
    preamble_bounds = {};
    preamble_iq = {};
    rssi = {};
    
    fprintf('Searching for %s\n', search_mac_tx);
    printout_counter = 0;
    for mac_i = 1:size(analyzer.Results, 2)
        mac_summary = macSummary(analyzer, mac_i, false);
        if isempty(mac_summary)
            continue;
        end
    
        % frame_rx_mac = parse_mac_address(mac_summary{1, 1}); % MAC address of the AP
        frame_tx_mac = parse_mac_address(mac_summary{1, 2}); % MAC address of the emitter
    
        % Filter frames based on the TX MAC address
        if strcmp(frame_tx_mac, search_mac_tx)
            fprintf('.');
            printout_counter = printout_counter + 1;
            if printout_counter == 40
                printout_counter = 0;
                fprintf('\n');
            end
    
            % Extract start & end indexes for the frame preamble
            x = analyzer.Results{mac_i};
    
            samples_start = x.PacketOffset;
        
            % Load full frame if preamble length isn't specified
            if preamble_len == -1
                samples_end = samples_start + x.NumRxSamples;
            else
                samples_end = samples_start + preamble_len - 1;
            end
    
            preamble_bounds{end+1} = [samples_start, samples_end];
            preamble_iq{end+1} = X(samples_start:samples_end);
            rssi{end+1} = round(10*log10(x.PacketPower),2);
        end
    end
    fprintf('\n');

    fprintf('Found %i TX frames.\n', length(preamble_bounds));

    T = struct();
    T.('preamble_bounds') = preamble_bounds;
    T.('preamble_iq') = preamble_iq;
    T.('rssi') = rssi;
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