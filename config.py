import optuna

from Controllers import BaseController
from Networks import (DalcaDiffNet, KrebsDiffNet, RBFDCGenerativeNetwork,
                      RBFGIGenerativeNetwork, RBFGIMutilRadiusAAdaptive,
                      VoxelMorph, RBFGIMutilRadiusAAdaptivePro)

config = {
    'mode': 'Train',
    'network': 'RBFDC',
    'name': 'Final-excludeMM',
    # All
    # 'dataset': {
    #     'training_list_path': 'G:\\cardiac_data\\training_pair.txt',
    #     'testing_list_path': 'G:\\cardiac_data\\testing_pair.txt',
    #     'validation_list_path': 'G:\\cardiac_data\\validation_pair.txt',
    #     'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    #     'resolution_path': 'G:\\cardiac_data\\resolution.txt'
    # },
    # ~MM
    'dataset': {
        'training_list_path': 'D:\\Data\\cardiac_data\\training_pair.txt',
        'testing_list_path': 'D:\\Data\\cardiac_data\\testing_pair.txt',
        'validation_list_path': 'D:\\Data\\cardiac_data\\validation_pair.txt',
        'pair_dir': 'D:\\Data\\cardiac_data\\2Dwithoutcenter1/',
        'resolution_path': 'D:\\Data\\cardiac_data\\resolution.txt'
    },
    # MM
    # 'dataset': {
    #     'training_list_path': 'G:\\cardiac_data\\M&M_training_pair.txt',
    #     'testing_list_path': 'G:\\cardiac_data\\M&M_testing_pair.txt',
    #     'validation_list_path': 'G:\\cardiac_data\\M&M_validation_pair.txt',
    #     'pair_dir': 'G:\\cardiac_data\\2Dwithoutcenter1/',
    #     'resolution_path': 'G:\\cardiac_data\\resolution.txt'
    # },
    'Train': {
        'batch_size': 32,
        'model_save_dir':
        'D:\\Code\\RegistrationPakageForNerualLearning\\modelForRadialPaper',
        'lr': 5e-4,
        'max_epoch': 3000,
        'save_checkpoint_step': 500,
        'v_step': 500,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 1000
        },
    },
    'Test': {
        'epoch': 'best',
        'model_save_path':
        'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\RBFGIMutilAdaptivePro\ECCV-all-WLCC---0.02-[9, 9]--1300001502-20220224115006',
        'excel_save_path': 'D:\Code\RegistrationPakageForNerualLearning\excel',
        'verbose': 2,
    },
    'SpeedTest': {
        'epoch': 'best',
        'model_save_path':
        'D:\Code\RegistrationPakageForNerualLearning\modelForRadialPaper\DalcaDiff\\0-MSE-50-11111.111111111111-20211108191421',
        'device': 'cpu'
    },
    'Hyperopt': {
        'n_trials': 30,
        'earlystop': {
            'min_delta': 0.00001,
            'patience': 500
        },
        'max_epoch': 800,
        'lr': 1e-4
    },
    'VoxelMorph': {
        'controller': BaseController,
        'network': VoxelMorph,
        'params': {
            'vol_size': [128, 128],
            'enc_nf': [16, 32, 32, 32],
            'dec_nf': [32, 32, 32, 32, 32, 16, 16],
            'reg_param': 1,
            'full_size': True,
            # WLCC
            # 'similarity_loss': 'WLCC',
            # 'similarity_loss_param': {
            #     'alpha': 0.02,
            #     'win': [9, 9]
            # }
            # LCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
            # MSE
            # 'similarity_loss': 'MSE',
            # 'similarity_loss_param': {
            #     'weight': 100
            # }
        }
    },
    'KrebsDiff': {
        'controller': BaseController,
        'network': KrebsDiffNet,
        'params': {
            'z_dim': 64,
            'encoder_param': {
                'num_layers': [0, 0, 0, 0],
                'dims': [16, 32, 32, 4],
                'last_block_dim': [],
            },
            'decoder_param': {
                'num_layers': [0, 0, 0],
                'dims': [32, 32, 32],
                'last_block_dim': [16]
            },
            'i_size': [128, 128],
            'similarity_factor': 60000,
            'smooth_kernel_size': 15,
            'smooth_sigma': 3,
            'factor': 4,
            # WLCC
            # 'similarity_loss': 'WLCC',
            # 'similarity_loss_param': {
            #     'alpha': 0.05,
            #     'win': [9, 9]
            # }
            # LCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            },
        }
    },
    'DalcaDiff': {
        'controller': BaseController,
        'network': DalcaDiffNet,
        'params': {
            'vol_size': [128, 128],
            'enc_nf': [16, 32, 32, 32],
            'dec_nf': [32, 32, 32, 32, 16, 3],
            # 'image_sigma': 0.03,
            'prior_lambda': 200000,
            # MSE
            'similarity_loss': 'MSE',
            'similarity_loss_param': {
                'weight': 1 / (0.03**2)  # 0.03 should be the image_sigma
            },
            # WLCC
            # 'similarity_loss': 'WLCC',
            # 'similarity_loss_param': {
            #     'alpha': 0.02,
            #     'win': [9, 9]
            # }
            # LCC
            # 'similarity_loss': 'LCC',
            # 'similarity_loss_param': {
            #     'win': [9, 9]
            # },
            'similarity_factor': 1,  # if MSE must be 1.
            'int_steps': 7,
            'vel_resize': 1 / 2,
            'bidir': False
        }
    },
    'RBFDC': {
        'controller': BaseController,
        'network': RBFDCGenerativeNetwork,
        'params': {
            'encoder_param': {
                'feature_dims': [16, 32, 32, 32, 32],
                'layer_nums': [2, 2, 2, 2, 2],
                'i_size': [128, 128]
            },
            'i_size': [128, 128],
            'c_factor': 2,
            'cpoint_num': 64,
            'align_corners': False,
            'factor_list': [160000, 0, 0],
            # WLCC
            # 'similarity_loss': 'WLCC',
            # 'similarity_loss_param': {
            #     'alpha': 0.02,
            #     'win': [9, 9]
            # }
            # LCC
            'similarity_loss': 'LCC',
            'similarity_loss_param': {
                'win': [9, 9]
            }
        },
        'hyperparams': {
            'factor_list': [
                {
                    'type': 'suggest_int',
                    'params': {
                        'low': 5000,
                        'high': 500000,
                        'step': 5000
                    }
                }, 0, 0
                #  {
                #     'type': 'suggest_int',
                #     'params': {
                #         'low': 0,
                #         'high': 1000,
                #         'step': 50
                #     }
                # }, {
                #     'type': 'suggest_float',
                #     'params': {
                #         'low': 0,
                #         'high': 10,
                #         'step': 0.5
                #     }
                # }
            ],
            #LCC
            'similarity_loss_param': {
                'win': [9, 9]
            }
        }
    },
    'RBFGI': {
        'controller': BaseController,
        'network': RBFGIGenerativeNetwork,
        'params': {
            'encoder_param': {
                'dims': [16, 32, 32, 32, 32],
                'num_layers': [2, 2, 2, 2, 2],
                'local_dims': [32, 32, 32, 32, 32],
                'local_num_layers': [2, 2, 2, 2, 2]
            },
            'c': 2,
            'i_size': [128, 128],
            'similarity_factor': 130000,
            # WLCC
            'similarity_loss': 'WLCC',
            'similarity_loss_param': {
                'alpha': 0.02,
                'win': [9, 9]
            }
            # LCC
            # 'similarity_loss': 'LCC',
            # 'similarity_loss_param': {
            #     'win': [9, 9]
            # },
        }
    },
    'RBFGIMutilAdaptive': {
        'controller': BaseController,
        'network': RBFGIMutilRadiusAAdaptive,
        'params': {
            'encoder_param': {
                'shared_param': {
                    'index': 4,
                    'dims': [16, 32, 32, 32, 32],
                    'num_layers': [2, 2, 2, 2, 2],
                    'local_dims': [32, 32, 32, 32, 32],
                    'local_num_layers': [2, 2, 2, 2, 2],
                },
                'unshared_param': {
                    'index': 5,
                    'dims': [16, 32, 32, 32, 32],
                    'num_layers': [2, 2, 2, 2, 2],
                    'local_dims': [32, 32, 32, 32, 32],
                    'local_num_layers': [2, 2, 2, 2, 2],
                },
                'c_nums': 3
            },
            'c_list': [1.5, 2, 2.5],
            'i_size': [128, 128],
            'factor_list': [130000, 0, 2],
            'int_steps': None,
            # WLCC
            'similarity_loss': 'WLCC',
            'similarity_loss_param': {
                'alpha': 0.02,
                'win': [9, 9]
            }
            # LCC
            # 'similarity_loss': 'LCC',
            # 'similarity_loss_param': {
            #     'win': [9, 9]
            # },
            # MSE
            # 'similarity_loss': 'MSE',
            # 'similarity_loss_param': {
            #     'weight': 1
            # }
        }
    },
    'RBFGIMutilAdaptivePro': {
        'controller': BaseController,
        'network': RBFGIMutilRadiusAAdaptivePro,
        'params': {
            'encoder_param': {
                'shared_param': {
                    'index': 4,
                    'dims': [16, 32, 32, 32, 32],
                    'num_layers': [2, 2, 2, 2, 2],
                    'local_dims': [32, 32, 32, 32, 32],
                    'local_num_layers': [2, 2, 2, 2, 2],
                },
                'unshared_param': {
                    'index': 5,
                    'dims': [16, 32, 32, 32, 32],
                    'num_layers': [2, 2, 2, 2, 2],
                    'local_dims': [32, 32, 32, 32, 32],
                    'local_num_layers': [2, 2, 2, 2, 2],
                },
                'c_nums': 3
            },
            'c_list': [1.5, 2, 2.5],
            'i_size': [128, 128],
            'factor_list': [130000, 150, 2],
            'int_steps': None,
            # WLCC
            'similarity_loss': 'WLCC',
            'similarity_loss_param': {
                'alpha': 0.02,
                'win': [9, 9]
            }
            # LCC
            # 'similarity_loss': 'LCC',
            # 'similarity_loss_param': {
            #     'win': [9, 9]
            # },
            # MSE
            # 'similarity_loss': 'MSE',
            # 'similarity_loss_param': {
            #     'weight': 1
            # }
        }
    }
}
