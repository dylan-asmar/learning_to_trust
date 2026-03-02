@testset "Constructor" begin
    @test TagMOMDPAT() isa TagMOMDPAT
    @test TagMOMDPAT(; tag_reward=20.0) isa TagMOMDPAT
    map_str = """
    xxxxxxxxxx
    xoooooooox
    xoxoxxxxox
    xoxoxxxxox
    xoxooooxox
    xoxoxxoxox
    xoxoxxoxox
    xoxoxxoxox
    xoooooooox
    xxxxxxxxxx
    """
    @test TagMOMDPAT(; map_str=map_str) isa TagMOMDPAT
    display(TagMOMDPAT(; map_str=map_str))
    @test discount(TagMOMDPAT()) == 0.95
    @test discount(TagMOMDPAT(; discount_factor=0.5)) == 0.5
    @test is_y_prime_dependent_on_x_prime(TagMOMDPAT()) == false
    @test is_x_prime_dependent_on_y(TagMOMDPAT()) == true
    @test is_initial_distribution_independent(TagMOMDPAT()) == true

    map_str = """xxx\nxox\nxyx\nooo"""
    @test_throws ErrorException TagMOMDPAT(; map_str=map_str)
    
    map_str = "xxx\nxox\noox\nooo\nxxox"
    @test_throws AssertionError TagMOMDPAT(; map_str=map_str)

    map_str = """xox\noxo\nooo\nxxx"""
    momdp = TagMOMDPAT(; map_str=map_str)
    @test get_prop(momdp.mg, :nrows) == 4
    @test get_prop(momdp.mg, :ncols) == 3
    @test get_prop(momdp.mg, :num_grid_pos) == 6
    @test ne(momdp.mg) == 8
    @test momdp.dist_matrix[1, 1] == 0.0
    @test isinf(momdp.dist_matrix[1, 2])
    @test isinf(momdp.dist_matrix[1, 3])
    @test isinf(momdp.dist_matrix[4, 1])
    @test momdp.dist_matrix[4, 3] == 3.0

    @test get_prop(momdp.mg, :node_pos_mapping)[1] == (1, 2)
    @test get_prop(momdp.mg, :node_pos_mapping)[2] == (2, 1)
    @test get_prop(momdp.mg, :node_pos_mapping)[4] == (3, 1)
    @test get_prop(momdp.mg, :node_mapping)[(3, 3)] == 6
    @test get_prop(momdp.mg, :node_mapping)[(2, 3)] == 3
    
    @test all(momdp.Q_ask_array .== 0)
    momdp = TagMOMDPAT(; 
        map_str=map_str,
        Q_ask_array=rand(7, 6, 5),
        num_asks=0
    )
    @test all(momdp.Q_ask_array .!= 0.0)
    @test_throws AssertionError TagMOMDPAT(; 
                                    map_str=map_str,
                                    Q_ask_array=rand(7, 6, 6),
                                    num_asks=0)
    @test_throws AssertionError TagMOMDPAT(; 
                                map_str=map_str,
                                Q_ask_array=rand(7, 7, 5),
                                num_asks=0)
    @test_throws AssertionError TagMOMDPAT(; 
                                map_str=map_str,
                                Q_ask_array=rand(6, 6, 5),
                                num_asks=0)
    momdp = TagMOMDPAT(; 
        map_str=map_str,
        Q_ask_array=rand(7, 6, 5),
        num_asks=1
    )
    @test all(momdp.Q_ask_array .!= 0.0)
    @test size(momdp.Q_ask_array)[3] == 5
    
    momdp = TagMOMDPAT(; num_asks=-1)
    @test momdp.num_asks == -1
    momdp = TagMOMDPAT(; num_asks=0)
    @test momdp.num_asks == 0
    @test_throws TypeError TagMOMDPAT(; num_asks=1.5)
    @test_throws AssertionError TagMOMDPAT(; num_asks=-2)
end
