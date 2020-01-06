load_package redlog;
rlset ofsf;
off nat;
off exp;
off echo;

dim := 20;


%%%%%%%%%%%%%%%%%%%%%nn parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

weight_in := mat((0.1148,  0.5470),
        ( 0.4275,  0.2984),
        (-0.1774,  0.0860),
        (-0.3760,  0.6816),
        ( 0.1487, -0.0492),
        ( 0.2377, -0.6829),
        ( 0.0392, -0.3617),
        ( 0.4332, -1.3175),
        ( 0.9111, -0.0625),
        (-0.0883,  0.7383),
        ( 0.5704,  0.1943),
        ( 0.4654, -0.5360),
        (-0.2125,  0.5549),
        ( 0.5454,  0.5400),
        (-0.3191, -0.0914),
        ( 0.5003,  0.4016),
        (-0.9437,  0.0942),
        (-0.5586, -0.1215),
        (-0.4198,  0.4475),
        (-0.4310, -0.5372));
bias_in := tp mat(( 0.1604, -0.4252,  0.5667,  0.1456, -0.2516,  0.3577,  0.6017,  0.1711,
        -0.5290,  0.1481, -0.5820, -0.6501,  0.9082,  1.0409,  0.2333, -0.0242,
        -0.7445, -0.9134,  0.1340, -0.1339));
weight_out := mat((0.3358, -0.2225, -0.1938, -0.3305, -0.0240,  0.5011, -0.8979,  0.1951,
         -0.5793,  0.4476, -0.1130,  1.0158,  0.6021, -0.5579, -0.6764,  0.6302,
          0.1109, -0.3937, -0.2109,  0.3426));
bias_out := mat((0.3856));


input_var := mat((x), (y)); % dim * 1
input_hidden := weight_in * input_var + bias_in; % dim * 1

output_hidden := mat((ho1), (ho2), (ho3), (ho4), (ho5), (ho6), (ho7), (ho8), (ho9), (ho10), (ho11), (ho12), (ho13), (ho14), (ho15), (ho16), (ho17), (ho18), (ho19), (ho20));
nn_output := weight_out * output_hidden + bias_out;
nn_output := nn_output(1, 1);

d_hidden := mat((dh1), (dh2), (dh3), (dh4), (dh5), (dh6), (dh7), (dh8), (dh9), (dh10), (dh11), (dh12), (dh13), (dh14), (dh15), (dh16), (dh17), (dh18), (dh19), (dh20) );

matrix m_diag(dim, dim);
for i:=1 step 1 until dim do << m_diag(i, i) := d_hidden(i, 1); >>;

dh_var := m_diag * weight_in; %% dim * 2
dnn_var := weight_out * dh_var; %% 1 * 2, gradient


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vector_field := mat((y, -x - y + x^3 / 3)); %% 1 * 2
init := (x - 1.5)^2 + y^2 <= 0.5^2 or (x >= -1.8 and x <= -1.2 and y >= -0.1 and y <= 0.1) or (x >= -1.4 and x <= -1.2 and y >= -0.5 and y <= 0.1);
unsafe := (x + 1)^2 + (y + 1)^2 <= 0.4^2 or (x >= 0.4 and x <= 0.6 and y >= 0.1 and y <= 0.5) or (x >= 0.4 and x <= 0.8 and y >= 0.1 and y <= 0.3);
domain := x >= -3 and x <= 2.5 and y >= -2 and y <= 1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lie := dnn_var * (tp vector_field); %% 1 * 2 * 2 * 1
lie := lie(1, 1);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

act_f := act_f_6;

app_act_f := true;
for i:=1 step 1 until dim do << app_act_f := app_act_f and sub({f_out=output_hidden(i, 1), f_in=input_hidden(i, 1)}, act_f); >>;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
deri_f :=  (f_in <= -0.1 and dh > 0 and dh < 0.00971) 
        or (f_in > -0.1 and f_in <= -0.05 and dh > 0.009709 and dh < 0.035762)
        or (f_in > -0.05 and f_in <= -0.02 and dh > 0.0724011 + 3.48087 * (0.033 + f_in) and dh < 0.0724011 + 3.48087 * (0.033 + f_in) + 0.0288 )
        or (f_in > -0.02 and f_in <= 0 and dh > 0.242752 + 15.7627 * (0.012 + f_in) and dh < 0.242752 + 15.7627 * (0.012 + f_in) + 0.0681 )
        or (f_in > 0 and f_in < 0.02 and dh > 0.757248 + 15.7627 * (-0.012 + f_in) - 0.0681 and dh < 0.757248 + 15.7627 * (-0.012 + f_in) )
        or (f_in >= 0.02 and f_in < 0.05 and dh > 0.927599 + 3.48087 * (-0.033 + f_in) - 0.0288 and dh < 0.927599 + 3.48087 * (-0.033 + f_in) )
        or (f_in >= 0.05 and f_in < 0.1 and dh > 0.9642 and dh < 0.9903)
        or (f_in >= 0.1 and dh < 1 and dh > 0.99028);

app_deri_f := true;
for i:=1 step 1 until dim do << app_deri_f := app_deri_f and sub({dh=d_hidden(i, 1), f_in=input_hidden(i, 1)}, deri_f); >>;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%verified using act_f_6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cons_init := init and app_act_f and nn_output >= 0$
cons_safe := unsafe and app_act_f and nn_output <= 0$
cons_lie := domain and app_act_f and nn_output = 0 and app_deri_f and lie >= 0;



out temp;
write cons_lie; %% act_f_6, msw 0.001
shut temp;
;END;

out temp;
write cons_safe; %% act_f_6, msw 0.001
shut temp;
;END;

out temp;
write cons_init; %% act_f_6, msw 0.001
shut temp;
;END;

