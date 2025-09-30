close all
format long

optimize_geometric_zetaN(10, 0.1, 4, 60, 0.1, [1.1 1.2 1.3 1.4 1.5 1.6 1.7 1.8 1.9 2.0 2.1 2.2])

optimize_zetaN(10, 0.1, 4, 60, 0)

function optimize_zetaN(N, E_min, E_max, T, dt)

    if (dt < 0)
        'choose dt > 0'
        stop
    end
    if (dt > 0)
        if (T/dt > floor(T/dt) + 1.E-10) 
            'T is not an integer multiple of dt'
            stop
        end
    end
    % Number of iterations for annealing
    maxIter = 2000;
    % Initial temperature
    T0 = 10.0;
    % Cooling rate
    cooling = 0.999;
    
    % Initial guess: equally divide T
    t = rand(1,N);
    t = t/sum(t)*T;
    if (dt > 0)
        k = ceil(rand*N);
        t = floor(t/dt)*dt;
        t(k) = T - sum(t([1:k-1 k+1:N]));
    end
    
    % Evaluate initial cost
    zeta_best = zetaN(t, E_min, E_max, N);
    t_best = t;
    
    T_curr = T0;
    
    for iter = 1:maxIter
        % Generate neighbor (perturbation that preserves sum t_i = T)
        t_new = t + max(0.1,2*dt) * (randn(1, N));
        t_new = abs(t_new); % make positive
        t_new = t_new * T / sum(t_new); % normalize to satisfy sum(t_new) = T
        if (dt > 0)
            k = ceil(rand*N);
            t_new = floor(t/dt)*dt;
            t_new(k) = T - sum(t_new([1:k-1 k+1:N]));
        end        

        % Compute new cost
        zeta_new = zetaN(t_new, E_min, E_max, N);
       
        % Accept or reject
        delta = zeta_new - zeta_best;
        if delta < 0 || rand < exp(-delta / T_curr)
            t = sort(t_new,'descend');
            if zeta_new < zeta_best
                zeta_best = zeta_new;
                t_best = t;
                [iter zeta_best]
                t
            end
        end
        
        % Cool down
        T_curr = T_curr * cooling;
    end
    
    % Display results
    fprintf('Minimum zeta_N found: %.6f\n', zeta_best);
    fprintf('Corresponding t_i values:\n');
    t_best

end


function optimize_geometric_zetaN(N, E_min, E_max, T, dt, alpha_array)

    if (dt < 0)
        'choose dt > 0'
        stop
    end
    if (dt == 0)
        nn = 0;
        for alpha = alpha_array 
            nn = nn+1;
            [nn size(alpha_array,2)]
            t = ((-1+alpha)*alpha^(-1+N)*T)/(-1+alpha^N)*1./(alpha.^[0:N-1]);
            zeta_array(nn) = zetaN(t, E_min, E_max, N);
        end
    end
    if (dt > 0)
        if (T/dt > floor(T/dt) + 1.E-10) 
            'T is not an integer multiple of dt'
            stop
        end
        nn = 0;
        for alpha = alpha_array 
            nn = nn+1;
            [nn size(alpha_array,2)]
            match = 0;
            Tcontinuum = T;
            while (match == 0)
                t = dt*round(((-1+alpha)*alpha^(-1+N)*Tcontinuum)/(-1+alpha^N)*1./(alpha.^[0:N-1]));
                ratio = sum(t)/T;
                if (abs(ratio - 1) > 1.E-10)                    
                    Tcontinuum = Tcontinuum/ratio;
                else
                    match = 1;
                end
            end
            zeta_array(nn) = zetaN(t, E_min, E_max, N);
        end        
    end    

    plot (alpha_array,log(zeta_array))
    xlabel('\alpha')
    ylabel('log(\zeta)')
    alpha_array
    zeta_array

end

function z = zetaN(t, E_min, E_max, N)

    % Generate all combinations of {-1, 1} for k_i and k_i'
    states = dec2bin(0:2^N-1) - '0';
    states = 2*states - 1; % Convert to {-1, 1}

    % slower version
    %
    % z_min = 0;
    % z_max = 0;
    % for i = 1:size(states,1)
    %     k = states(i,:);
    %     for j = 1:size(states,1)
    %         kp = states(j,:);
    %         sum_min_term = sum((k + kp) .* t) * E_min / 2;
    %         sum_max_term = sum((k + kp) .* t) * E_max / 2;            
    %         z_min = z_min + sinc(sum_min_term);
    %         z_max = z_max + sinc(sum_max_term);                    
    %     end
    % end 

    M = size(states, 1);
    z_min = 0;
    z_max = 0;
    for i = 1:M
        k = states(i, :);                     % 1 x d
        k_sum = states + k;                   % M x d (adds k to every row of states)
        dot_vals = k_sum * t(:);              % M x 1
        z_min = z_min + sum(sinc(dot_vals * E_min / 2));
        z_max = z_max + sum(sinc(dot_vals * E_max / 2));
    end

    z = (E_max * z_max - E_min * z_min) / (2^(2*N - 1));

end