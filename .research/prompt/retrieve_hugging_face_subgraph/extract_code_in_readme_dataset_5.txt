
Input:
From the Hugging Face README provided in “# README,” extract and output only the Python code required for execution. Do not output any other information. In particular, if no implementation method is described, output an empty string.

# README
---
configs:
- config_name: default
  data_files:
  - split: cifar100_2
    path: data/cifar100_2-*
  - split: cifar100_3
    path: data/cifar100_3-*
  - split: cifar100_4
    path: data/cifar100_4-*
  - split: cifar100_5
    path: data/cifar100_5-*
  - split: cifar100_6
    path: data/cifar100_6-*
  - split: cifar100_7
    path: data/cifar100_7-*
  - split: cifar100_8
    path: data/cifar100_8-*
  - split: cifar100_9
    path: data/cifar100_9-*
  - split: cifar100_10
    path: data/cifar100_10-*
  - split: cifar100_11
    path: data/cifar100_11-*
  - split: cifar100_12
    path: data/cifar100_12-*
  - split: cifar100_13
    path: data/cifar100_13-*
  - split: cifar100_14
    path: data/cifar100_14-*
  - split: cifar100_15
    path: data/cifar100_15-*
  - split: cifar100_16
    path: data/cifar100_16-*
  - split: cifar100_17
    path: data/cifar100_17-*
  - split: cifar100_18
    path: data/cifar100_18-*
  - split: cifar100_19
    path: data/cifar100_19-*
  - split: cifar100_20
    path: data/cifar100_20-*
  - split: cifar100_21
    path: data/cifar100_21-*
  - split: cifar100_22
    path: data/cifar100_22-*
  - split: cifar100_23
    path: data/cifar100_23-*
  - split: cifar100_24
    path: data/cifar100_24-*
  - split: cifar100_25
    path: data/cifar100_25-*
  - split: cifar100_26
    path: data/cifar100_26-*
  - split: cifar100_27
    path: data/cifar100_27-*
  - split: cifar100_28
    path: data/cifar100_28-*
  - split: cifar100_29
    path: data/cifar100_29-*
  - split: cifar100_30
    path: data/cifar100_30-*
  - split: cifar100_31
    path: data/cifar100_31-*
  - split: cifar100_32
    path: data/cifar100_32-*
  - split: cifar100_33
    path: data/cifar100_33-*
  - split: cifar100_34
    path: data/cifar100_34-*
  - split: cifar100_35
    path: data/cifar100_35-*
  - split: cifar100_36
    path: data/cifar100_36-*
  - split: cifar100_37
    path: data/cifar100_37-*
  - split: cifar100_38
    path: data/cifar100_38-*
  - split: cifar100_39
    path: data/cifar100_39-*
  - split: cifar100_40
    path: data/cifar100_40-*
  - split: cifar100_41
    path: data/cifar100_41-*
  - split: cifar100_42
    path: data/cifar100_42-*
  - split: cifar100_43
    path: data/cifar100_43-*
  - split: cifar100_44
    path: data/cifar100_44-*
  - split: cifar100_45
    path: data/cifar100_45-*
  - split: cifar100_46
    path: data/cifar100_46-*
  - split: cifar100_47
    path: data/cifar100_47-*
  - split: cifar100_48
    path: data/cifar100_48-*
  - split: cifar100_49
    path: data/cifar100_49-*
  - split: cifar100_50
    path: data/cifar100_50-*
  - split: cifar100_51
    path: data/cifar100_51-*
  - split: cifar100_52
    path: data/cifar100_52-*
  - split: cifar100_53
    path: data/cifar100_53-*
  - split: cifar100_54
    path: data/cifar100_54-*
  - split: cifar100_55
    path: data/cifar100_55-*
  - split: cifar100_56
    path: data/cifar100_56-*
  - split: cifar100_57
    path: data/cifar100_57-*
  - split: cifar100_58
    path: data/cifar100_58-*
  - split: cifar100_59
    path: data/cifar100_59-*
  - split: cifar100_60
    path: data/cifar100_60-*
  - split: cifar100_61
    path: data/cifar100_61-*
  - split: cifar100_62
    path: data/cifar100_62-*
  - split: cifar100_63
    path: data/cifar100_63-*
  - split: cifar100_64
    path: data/cifar100_64-*
  - split: cifar100_65
    path: data/cifar100_65-*
  - split: cifar100_66
    path: data/cifar100_66-*
  - split: cifar100_67
    path: data/cifar100_67-*
  - split: cifar100_68
    path: data/cifar100_68-*
  - split: cifar100_69
    path: data/cifar100_69-*
  - split: cifar100_70
    path: data/cifar100_70-*
  - split: cifar100_71
    path: data/cifar100_71-*
  - split: cifar100_72
    path: data/cifar100_72-*
  - split: cifar100_73
    path: data/cifar100_73-*
  - split: cifar100_74
    path: data/cifar100_74-*
  - split: cifar100_75
    path: data/cifar100_75-*
  - split: cifar100_76
    path: data/cifar100_76-*
  - split: cifar100_77
    path: data/cifar100_77-*
  - split: cifar100_78
    path: data/cifar100_78-*
  - split: cifar100_79
    path: data/cifar100_79-*
  - split: cifar100_80
    path: data/cifar100_80-*
  - split: cifar100_81
    path: data/cifar100_81-*
  - split: cifar100_82
    path: data/cifar100_82-*
  - split: cifar100_83
    path: data/cifar100_83-*
  - split: cifar100_84
    path: data/cifar100_84-*
  - split: cifar100_85
    path: data/cifar100_85-*
  - split: cifar100_86
    path: data/cifar100_86-*
  - split: cifar100_87
    path: data/cifar100_87-*
  - split: cifar100_88
    path: data/cifar100_88-*
  - split: cifar100_89
    path: data/cifar100_89-*
  - split: cifar100_90
    path: data/cifar100_90-*
  - split: cifar100_91
    path: data/cifar100_91-*
  - split: cifar100_92
    path: data/cifar100_92-*
  - split: cifar100_93
    path: data/cifar100_93-*
  - split: cifar100_94
    path: data/cifar100_94-*
  - split: cifar100_95
    path: data/cifar100_95-*
  - split: cifar100_96
    path: data/cifar100_96-*
  - split: cifar100_97
    path: data/cifar100_97-*
  - split: cifar100_98
    path: data/cifar100_98-*
  - split: cifar100_99
    path: data/cifar100_99-*
  - split: cifar100_100
    path: data/cifar100_100-*
dataset_info:
  features:
  - name: img
    dtype: image
  - name: fine_label
    dtype:
      class_label:
        names:
          '0': apple
          '1': aquarium_fish
          '2': baby
          '3': bear
          '4': beaver
          '5': bed
          '6': bee
          '7': beetle
          '8': bicycle
          '9': bottle
          '10': bowl
          '11': boy
          '12': bridge
          '13': bus
          '14': butterfly
          '15': camel
          '16': can
          '17': castle
          '18': caterpillar
          '19': cattle
          '20': chair
          '21': chimpanzee
          '22': clock
          '23': cloud
          '24': cockroach
          '25': couch
          '26': cra
          '27': crocodile
          '28': cup
          '29': dinosaur
          '30': dolphin
          '31': elephant
          '32': flatfish
          '33': forest
          '34': fox
          '35': girl
          '36': hamster
          '37': house
          '38': kangaroo
          '39': keyboard
          '40': lamp
          '41': lawn_mower
          '42': leopard
          '43': lion
          '44': lizard
          '45': lobster
          '46': man
          '47': maple_tree
          '48': motorcycle
          '49': mountain
          '50': mouse
          '51': mushroom
          '52': oak_tree
          '53': orange
          '54': orchid
          '55': otter
          '56': palm_tree
          '57': pear
          '58': pickup_truck
          '59': pine_tree
          '60': plain
          '61': plate
          '62': poppy
          '63': porcupine
          '64': possum
          '65': rabbit
          '66': raccoon
          '67': ray
          '68': road
          '69': rocket
          '70': rose
          '71': sea
          '72': seal
          '73': shark
          '74': shrew
          '75': skunk
          '76': skyscraper
          '77': snail
          '78': snake
          '79': spider
          '80': squirrel
          '81': streetcar
          '82': sunflower
          '83': sweet_pepper
          '84': table
          '85': tank
          '86': telephone
          '87': television
          '88': tiger
          '89': tractor
          '90': train
          '91': trout
          '92': tulip
          '93': turtle
          '94': wardrobe
          '95': whale
          '96': willow_tree
          '97': wolf
          '98': woman
          '99': worm
  - name: coarse_label
    dtype:
      class_label:
        names:
          '0': aquatic_mammals
          '1': fish
          '2': flowers
          '3': food_containers
          '4': fruit_and_vegetables
          '5': household_electrical_devices
          '6': household_furniture
          '7': insects
          '8': large_carnivores
          '9': large_man-made_outdoor_things
          '10': large_natural_outdoor_scenes
          '11': large_omnivores_and_herbivores
          '12': medium_mammals
          '13': non-insect_invertebrates
          '14': people
          '15': reptiles
          '16': small_mammals
          '17': trees
          '18': vehicles_1
          '19': vehicles_2
  splits:
  - name: cifar100_2
    num_bytes: 2250027.12
    num_examples: 1000
  - name: cifar100_3
    num_bytes: 3375040.68
    num_examples: 1500
  - name: cifar100_4
    num_bytes: 4500054.24
    num_examples: 2000
  - name: cifar100_5
    num_bytes: 5625067.8
    num_examples: 2500
  - name: cifar100_6
    num_bytes: 6750081.36
    num_examples: 3000
  - name: cifar100_7
    num_bytes: 7875094.92
    num_examples: 3500
  - name: cifar100_8
    num_bytes: 9000108.48
    num_examples: 4000
  - name: cifar100_9
    num_bytes: 10125122.04
    num_examples: 4500
  - name: cifar100_10
    num_bytes: 11250135.6
    num_examples: 5000
  - name: cifar100_11
    num_bytes: 12375149.16
    num_examples: 5500
  - name: cifar100_12
    num_bytes: 13500162.72
    num_examples: 6000
  - name: cifar100_13
    num_bytes: 14625176.28
    num_examples: 6500
  - name: cifar100_14
    num_bytes: 15750189.84
    num_examples: 7000
  - name: cifar100_15
    num_bytes: 16875203.4
    num_examples: 7500
  - name: cifar100_16
    num_bytes: 18000216.96
    num_examples: 8000
  - name: cifar100_17
    num_bytes: 19125230.52
    num_examples: 8500
  - name: cifar100_18
    num_bytes: 20250244.08
    num_examples: 9000
  - name: cifar100_19
    num_bytes: 21375257.64
    num_examples: 9500
  - name: cifar100_20
    num_bytes: 22500271.2
    num_examples: 10000
  - name: cifar100_21
    num_bytes: 23625284.76
    num_examples: 10500
  - name: cifar100_22
    num_bytes: 24750298.32
    num_examples: 11000
  - name: cifar100_23
    num_bytes: 25875311.88
    num_examples: 11500
  - name: cifar100_24
    num_bytes: 27000325.44
    num_examples: 12000
  - name: cifar100_25
    num_bytes: 28125339.0
    num_examples: 12500
  - name: cifar100_26
    num_bytes: 29250352.56
    num_examples: 13000
  - name: cifar100_27
    num_bytes: 30375366.12
    num_examples: 13500
  - name: cifar100_28
    num_bytes: 31500379.68
    num_examples: 14000
  - name: cifar100_29
    num_bytes: 32625393.24
    num_examples: 14500
  - name: cifar100_30
    num_bytes: 33750406.8
    num_examples: 15000
  - name: cifar100_31
    num_bytes: 34875420.36
    num_examples: 15500
  - name: cifar100_32
    num_bytes: 36000433.92
    num_examples: 16000
  - name: cifar100_33
    num_bytes: 37125447.48
    num_examples: 16500
  - name: cifar100_34
    num_bytes: 38250461.04
    num_examples: 17000
  - name: cifar100_35
    num_bytes: 39375474.6
    num_examples: 17500
  - name: cifar100_36
    num_bytes: 40500488.16
    num_examples: 18000
  - name: cifar100_37
    num_bytes: 41625501.72
    num_examples: 18500
  - name: cifar100_38
    num_bytes: 42750515.28
    num_examples: 19000
  - name: cifar100_39
    num_bytes: 43875528.84
    num_examples: 19500
  - name: cifar100_40
    num_bytes: 45000542.4
    num_examples: 20000
  - name: cifar100_41
    num_bytes: 46125555.96
    num_examples: 20500
  - name: cifar100_42
    num_bytes: 47250569.52
    num_examples: 21000
  - name: cifar100_43
    num_bytes: 48375583.08
    num_examples: 21500
  - name: cifar100_44
    num_bytes: 49500596.64
    num_examples: 22000
  - name: cifar100_45
    num_bytes: 50625610.2
    num_examples: 22500
  - name: cifar100_46
    num_bytes: 51750623.76
    num_examples: 23000
  - name: cifar100_47
    num_bytes: 52875637.32
    num_examples: 23500
  - name: cifar100_48
    num_bytes: 54000650.88
    num_examples: 24000
  - name: cifar100_49
    num_bytes: 55125664.44
    num_examples: 24500
  - name: cifar100_50
    num_bytes: 56250678.0
    num_examples: 25000
  - name: cifar100_51
    num_bytes: 57375691.56
    num_examples: 25500
  - name: cifar100_52
    num_bytes: 58500705.12
    num_examples: 26000
  - name: cifar100_53
    num_bytes: 59625718.68
    num_examples: 26500
  - name: cifar100_54
    num_bytes: 60750732.24
    num_examples: 27000
  - name: cifar100_55
    num_bytes: 61875745.8
    num_examples: 27500
  - name: cifar100_56
    num_bytes: 63000759.36
    num_examples: 28000
  - name: cifar100_57
    num_bytes: 64125772.92
    num_examples: 28500
  - name: cifar100_58
    num_bytes: 65250786.48
    num_examples: 29000
  - name: cifar100_59
    num_bytes: 66375800.04
    num_examples: 29500
  - name: cifar100_60
    num_bytes: 67500813.6
    num_examples: 30000
  - name: cifar100_61
    num_bytes: 68625827.16
    num_examples: 30500
  - name: cifar100_62
    num_bytes: 69750840.72
    num_examples: 31000
  - name: cifar100_63
    num_bytes: 70875854.28
    num_examples: 31500
  - name: cifar100_64
    num_bytes: 72000867.84
    num_examples: 32000
  - name: cifar100_65
    num_bytes: 73125881.4
    num_examples: 32500
  - name: cifar100_66
    num_bytes: 74250894.96
    num_examples: 33000
  - name: cifar100_67
    num_bytes: 75375908.52
    num_examples: 33500
  - name: cifar100_68
    num_bytes: 76500922.08
    num_examples: 34000
  - name: cifar100_69
    num_bytes: 77625935.64
    num_examples: 34500
  - name: cifar100_70
    num_bytes: 78750949.2
    num_examples: 35000
  - name: cifar100_71
    num_bytes: 79875962.76
    num_examples: 35500
  - name: cifar100_72
    num_bytes: 81000976.32
    num_examples: 36000
  - name: cifar100_73
    num_bytes: 82125989.88
    num_examples: 36500
  - name: cifar100_74
    num_bytes: 83251003.44
    num_examples: 37000
  - name: cifar100_75
    num_bytes: 84376017.0
    num_examples: 37500
  - name: cifar100_76
    num_bytes: 85501030.56
    num_examples: 38000
  - name: cifar100_77
    num_bytes: 86626044.12
    num_examples: 38500
  - name: cifar100_78
    num_bytes: 87751057.68
    num_examples: 39000
  - name: cifar100_79
    num_bytes: 88876071.24
    num_examples: 39500
  - name: cifar100_80
    num_bytes: 90001084.8
    num_examples: 40000
  - name: cifar100_81
    num_bytes: 91126098.36
    num_examples: 40500
  - name: cifar100_82
    num_bytes: 92251111.92
    num_examples: 41000
  - name: cifar100_83
    num_bytes: 93376125.48
    num_examples: 41500
  - name: cifar100_84
    num_bytes: 94501139.04
    num_examples: 42000
  - name: cifar100_85
    num_bytes: 95626152.6
    num_examples: 42500
  - name: cifar100_86
    num_bytes: 96751166.16
    num_examples: 43000
  - name: cifar100_87
    num_bytes: 97876179.72
    num_examples: 43500
  - name: cifar100_88
    num_bytes: 99001193.28
    num_examples: 44000
  - name: cifar100_89
    num_bytes: 100126206.84
    num_examples: 44500
  - name: cifar100_90
    num_bytes: 101251220.4
    num_examples: 45000
  - name: cifar100_91
    num_bytes: 102376233.96
    num_examples: 45500
  - name: cifar100_92
    num_bytes: 103501247.52
    num_examples: 46000
  - name: cifar100_93
    num_bytes: 104626261.08
    num_examples: 46500
  - name: cifar100_94
    num_bytes: 105751274.64
    num_examples: 47000
  - name: cifar100_95
    num_bytes: 106876288.2
    num_examples: 47500
  - name: cifar100_96
    num_bytes: 108001301.76
    num_examples: 48000
  - name: cifar100_97
    num_bytes: 109126315.32
    num_examples: 48500
  - name: cifar100_98
    num_bytes: 110251328.88
    num_examples: 49000
  - name: cifar100_99
    num_bytes: 111376342.44
    num_examples: 49500
  - name: cifar100_100
    num_bytes: 112501356.0
    num_examples: 50000
  download_size: 5989828624
  dataset_size: 5680193464.44
---
# Dataset Card for "cifar100_2_to_100"

[More Information needed](https://github.com/huggingface/datasets/blob/main/CONTRIBUTING.md#how-to-contribute-to-the-dataset-cards)
Output:
{
    "extracted_code": ""
}
