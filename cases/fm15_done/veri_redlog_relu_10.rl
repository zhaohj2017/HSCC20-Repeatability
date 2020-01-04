load_package redlog;
rlset ofsf;
off nat;
off exp;
off echo;

dim := 10;

%%%%%%%%%%%%%%%%%%%%%nn parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weight_in := mat((-0.4653,  0.1081),
        (-0.2802, -0.8425),
        ( 0.3592,  0.2013),
        (-0.1568,  2.2246),
        (-0.3086, -1.4097),
        ( 1.4852, -0.9569),
        ( 0.2094,  0.1350),
        (-0.0448, -0.6343),
        ( 0.9277,  0.2325),
        ( 0.1044, -0.3599));
bias_in := tp mat((-1.1673,  0.4850,  0.1627, -0.9033, -0.6401, -0.4655,  0.1198, -0.4248,
        -0.7313,  0.4120));
weight_out := mat((0.2336,  0.4914, -0.0023,  0.4555, -0.2802,  0.5934,  0.0051, -0.2958,
          0.6572, -0.3827));
bias_out := mat((-0.5387));


input_var := mat((x), (y)); % 2 * 1
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

act_f := act_f_6;


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

dh_var := m_diag * weight_in; %% dim * 2

dnn_var := weight_out * dh_var; %% 1 * 2

vector_field := mat((exp(-x) + y - 1, -sin(x) * sin(x))); %% 1 * 2

lie := dnn_var * (tp vector_field);
lie := lie(1, 1);
lie := part(lie, 1);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
init := (x + 0.5)^2 + (y - 0.5)^2 <= 0.16;
unsafe := (x - 0.7)^2 + (y + 0.7)^2 <= 0.09;
domain := x >= -2 and x <= 2 and y >= -2 and y <= 2;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%verified using act_f_6%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cons_init := init and app_act_f and nn_output >= 0$
cons_safe := unsafe and app_act_f and nn_output <= 0$


out temp$
write cons_safe;
shut temp$
write "Output cons_safe done!"; %% msw 0.01

;END;



out temp$
write cons_init;
shut temp$
write "Output cons_init done!"; %% msw 0.1

;END;






%%%%%%%%%%%%%%%%%%%%%%% boundary: veri_lie using act_f_6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
veri_lie := domain and app_act_f and nn_output = 0 and app_deri_f and lie >= 0;

out temp$
write veri_lie$
shut temp$
write "Output veri_lie done!"; %% msw 0.001

;END;



