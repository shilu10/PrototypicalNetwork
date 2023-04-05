iMAX_EPOCHS = 5
EPOCH_SIZE = 500

proto_model = load_protonet_conv(
    x_dim=(3,28,28),
    hid_dim=128,
    z_dim=64,
)
optimizer = optim.Adam(proto_model.parameters(), lr = 0.0001)


n_ways = [50, 10]
n_supports = [5, 50]
n_queries = [5]


for _ in range(len(n_ways)): 
    for n_way in n_ways:
        for n_support in n_supports: 
            for n_query in n_queries:
                print(f"n_way: {n_way}, n_support: {n_support}, n_query: {n_query}")
                proto_trainer = PrototypicalModel(proto_model, optimizer)
                trained_model = proto_trainer.train(trainx, trainy, n_way, n_support, 
                                                    n_query, MAX_EPOCHS, EPOCH_SIZE)
                torch.save(trained_model.state_dict(), f"models/proto_trained_model_{n_way}_n_way_{n_support}_k_shot")
                print("-----------------------------------------------------------")
