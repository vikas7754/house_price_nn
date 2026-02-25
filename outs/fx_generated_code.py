


def forward(self, x):
    base_model_net_0 = getattr(self.base_model.net, "0")(x);  x = None
    base_model_net_1 = getattr(self.base_model.net, "1")(base_model_net_0);  base_model_net_0 = None
    base_model_net_2 = getattr(self.base_model.net, "2")(base_model_net_1);  base_model_net_1 = None
    base_model_net_3 = getattr(self.base_model.net, "3")(base_model_net_2);  base_model_net_2 = None
    base_model_net_4 = getattr(self.base_model.net, "4")(base_model_net_3);  base_model_net_3 = None
    scale_tensor = self.scale_tensor
    custom_mul = torch.ops.my_mul_ops.custom_mul(base_model_net_4, scale_tensor);  base_model_net_4 = scale_tensor = None
    bias_tensor = self.bias_tensor
    custom_add = torch.ops.my_ops.custom_add(custom_mul, bias_tensor);  custom_mul = bias_tensor = None
    return custom_add
    