load_package redlog;
rlset ofsf;
off nat;
off exp;
off echo;

dim := 10;

%%%%%%%%%%%%%%%%%%%%%nn parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weight_in := mat((-0.0154,  0.1207,  0.0721),
        ( 0.0855,  1.5956,  0.2696),
        (-0.9973, -0.9949, -0.4558),
        ( 0.9997,  0.0550,  0.8147),
        ( 0.2742,  1.2206, -0.7480),
        (-0.2475,  0.2890, -0.0539),
        (-1.0115,  0.3225, -0.2749),
        ( 0.3411,  0.4940, -0.5462),
        ( 0.0259, -0.0483,  0.8617),
        (-0.6153,  1.1355,  0.1721));
bias_in := tp mat((-0.5217,  0.4236, -0.3542, -1.2463, -0.2801, -0.6170, -0.1140,  0.2003,
         0.5919,  0.3090));
weight_out := mat((-0.1039, -0.6013, -1.0555, -0.5538,  0.5712, -0.6276, -0.4307,  0.3405,
          0.6893, -0.4951));
bias_out := mat((0.3317));


input_var := mat((x), (y), (z)); % 3 * 1
input_hidden := weight_in * input_var + bias_in; % dim * 1

%activation function
%x = 0.5 * x + torch.sqrt(0.25 * x * x + 0.0001)

act_f_6 := (f_in <= -0.1 and f_out > 0 and f_out < 0.0009902)
        or (f_in > -0.1 and f_in <= -0.05 and f_out > 0.0009901 and f_out < 0.001926)
        or (f_in > -0.05 and f_in <= 0 and f_out > 0.00414214 + 0.146447 * (0.02 + f_in) - 0.00000001 and f_out < 0.00414214 + 0.146447 * (0.02 + f_in) +  0.00293 and f_out > 0) 
        or (f_in > 0 and f_in < 0.05 and f_out > 0.0224536 + 0.834482 * (-0.018 + f_in) and f_out < 0.0224536 + 0.834482 * (-0.018 + f_in) + 0.00277) 
        or (f_in >= 0.05 and f_in < 0.1 and f_out > f_in + 0.0009901 and f_out < f_in + 0.001926)
        or (f_in >= 0.1 and f_out > f_in and f_out < f_in + 0.0009902); 

act_f_4 := (f_in <= -0.05 and f_out > 0 and f_out < 0.001926)
        or (f_in > -0.05 and f_in <= 0 and f_out > 0.001925 and f_out <= 0.01)
        or (f_in > 0 and f_in < 0.05 and f_out > f_in + 0.001925 and f_out < f_in + 0.01)
        or (f_in >= 0.05 and f_out > f_in and f_out < f_in + 0.001926);

act_f_2 := (f_in <= 0 and f_out > 0 and f_out <= 0.01)
        or (f_in > 0 and f_out > f_in and f_out < f_in + 0.01);

act_f := act_f_2;

output_hidden := mat((ho1), (ho2), (ho3), (ho4), (ho5), (ho6), (ho7), (ho8), (ho9), (ho10));

app_act_f := true;
for i:=1 step 1 until dim do << app_act_f := app_act_f and sub({f_out=output_hidden(i, 1), f_in=input_hidden(i, 1)}, act_f); >>;

nn_output := weight_out * output_hidden + bias_out;
nn_output := nn_output(1, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
deri_f :=  (f_in <= -0.1 and dh > 0 and dh < 0.00971) 
        or (f_in > -0.1 and f_in <= -0.05 and dh > 0.009709 and dh < 0.035762)
        or (f_in > -0.05 and f_in <= -0.02 and dh > 0.0724011 + 3.48087 * (0.033 + f_in) and dh < 0.0724011 + 3.48087 * (0.033 + f_in) + 0.0288 )
        or (f_in > -0.02 and f_in <= 0 and dh > 0.242752 + 15.7627 * (0.012 + f_in) and dh < 0.242752 + 15.7627 * (0.012 + f_in) + 0.0681 )
        or (f_in > 0 and f_in < 0.02 and dh > 0.757248 + 15.7627 * (-0.012 + f_in) - 0.0681 and dh < 0.757248 + 15.7627 * (-0.012 + f_in) )
        or (f_in >= 0.02 and f_in < 0.05 and dh > 0.927599 + 3.48087 * (-0.033 + f_in) - 0.0288 and dh < 0.927599 + 3.48087 * (-0.033 + f_in) )
        or (f_in >= 0.05 and f_in < 0.1 and dh > 0.9642 and dh < 0.9903)
        or (f_in >= 0.1 and dh < 1 and dh > 0.99028);


d_hidden := mat((dh1), (dh2), (dh3), (dh4), (dh5), (dh6), (dh7), (dh8), (dh9), (dh10));

app_deri_f := true;
for i:=1 step 1 until dim do << app_deri_f := app_deri_f and sub({dh=d_hidden(i, 1), f_in=input_hidden(i, 1)}, deri_f); >>;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
matrix m_diag(dim, dim);
for i:=1 step 1 until dim do << m_diag(i, i) := d_hidden(i, 1); >>;

dh_var := m_diag * weight_in; %% dim * 3

dnn_var := weight_out * dh_var; %% 1 * 3, gradient

vector_field := mat((sin(z), cos(z), -sin(z) + 3 * (x * sin(z) + y * cos(z)) / (0.5 + x^2 + y^2) )); %% 1 * 3

lie := dnn_var * (tp vector_field); %% 1 * 3 * 3 * 1
lie := lie(1, 1);
lie := part(lie, 1);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
init := x >= -0.1 and x <= 0.1 and y >= -2 and y <= -1.8 and z >= -0.52 and z <= 0.52;
unsafe := x^2 + y^2 <= 0.04;
domain := x >= -2 and x <= 2 and y >= -2 and y <= 2 and z >= -1.57 and z <= 1.57;



%%%%%%%%%%%%%%%%%%%%%%%reduced boundary: veri_lie using act_f_2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
veri_lie := domain and app_act_f and nn_output = 0 and app_deri_f and lie >= 0;

out temp$
write veri_lie$
shut temp$
write "Output veri_lie done!"; %% msw 0.001: why not use act_f_2???
                                %% yes, using act_f_2 costs less time

;END;




%%%%%%%%%%%%%%%%%%%%%%%%%%using act_f_6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cons_init := init and app_act_f and nn_output >= 0;
cons_safe := unsafe and app_act_f and nn_output <= 0;

out temp$
write cons_safe$
shut temp$
write "Output cons_safe done!"; %% msw 0.001

;END;


out temp$
write cons_init$
shut temp$
write "Output cons_init done!"; %% msw 0.1

;END;


