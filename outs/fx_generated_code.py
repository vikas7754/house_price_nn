


def forward(self, x):
    net_0 = getattr(self.net, "0")(x);  x = None
    net_1 = getattr(self.net, "1")(net_0);  net_0 = None
    net_2 = getattr(self.net, "2")(net_1);  net_1 = None
    net_3 = getattr(self.net, "3")(net_2);  net_2 = None
    net_4 = getattr(self.net, "4")(net_3);  net_3 = None
    return net_4
    