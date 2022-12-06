import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []

    inF = open(bvh_file_path, 'r')
    inLines = inF.read().splitlines()
    parentIdStack = []
    parentIdStack.append(-1)
    
    curBoneIdx = -1
    for line in inLines:
        if 'ROOT' in line or 'JOINT' in line or 'End' in line:
            curBoneIdx += 1
            parentId = parentIdStack.pop()
            parentIdStack.append(parentId)
            joint_parent.append(parentId)
            curData = line.strip()
            curData = curData.split()
            if 'End' in line:
                joint_name.append(joint_name[parentId]+'_end')
            else:
                joint_name.append(curData[-1])
            joint_offset.append([0,0,0]) # initialization
            continue
        if '{' in line:
            parentIdStack.append(curBoneIdx)
            continue
        if 'OFFSET' in line:
            curData = line.strip()
            curData = curData.split()
            joint_offset[curBoneIdx][0] = float(curData[1])
            joint_offset[curBoneIdx][1] = float(curData[2])
            joint_offset[curBoneIdx][2] = float(curData[3])
            continue
        if '}' in line:
            parentIdStack.pop()
            continue

    inF.close()
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    curMotionData = motion_data[frame_id]
    boneNum = len(joint_name)
    joint_positions = np.zeros((boneNum, 3))
    joint_orientations = np.zeros((boneNum, 4))

    ## 假定bvh只包含一个root
    rootOffset = curMotionData[:3]
    for t in range(3):
        joint_positions[0][t] = joint_offset[0][t] + rootOffset[t]
    rootRotation = R.from_euler('XYZ', curMotionData[3:6], degrees=True).as_quat()
    joint_orientations[0] = (rootRotation)
    dataIdx = 1
    for bidx in range(1, boneNum):
        if '_end' in joint_name[bidx]:
            curRotation = R.from_euler('XYZ', [0,0,0], degrees=True)
        else:
            curRotation = R.from_euler('XYZ', curMotionData[3+dataIdx*3:6+dataIdx*3], degrees=True)
            dataIdx += 1
        parentBidx = joint_parent[bidx]
        parentOrientation = joint_orientations[parentBidx]
        curOrientation = R.from_quat(parentOrientation).as_matrix() @ curRotation.as_matrix()
        joint_orientations[bidx] = (R.from_matrix(curOrientation).as_quat())
        # joint_positions[bidx] = curOrientation @ joint_offset[bidx] + joint_positions[parentBidx]
        joint_positions[bidx] = R.from_quat(parentOrientation).as_matrix() @ joint_offset[bidx] + joint_positions[parentBidx]

    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    t_joint_name, t_joint_parent, t_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    a_joint_name, a_joint_parent, a_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    a_motion_data = load_motion_data(A_pose_bvh_path)

    a_joint_idx = {}
    idx = 0
    for item in a_joint_name:
        if '_end' not in item:
            a_joint_idx[item] = idx
            idx += 1
    t_joint_a_idx = {}
    for item in t_joint_name:
        if '_end' not in item:
            t_joint_a_idx[item] = a_joint_idx[item]
    
    motion_data = []
    for line in a_motion_data:
        cur_data = []
        cur_data.append(line[0])
        cur_data.append(line[1])
        cur_data.append(line[2])
        for item in t_joint_name:
            if '_end' not in item:
                cur_a_idx = t_joint_a_idx[item]
                if item == 'lShoulder':
                    cur_data.append(line[3+cur_a_idx*3+0])
                    cur_data.append(line[3+cur_a_idx*3+1])
                    cur_data.append(line[3+cur_a_idx*3+2]-45)
                elif item == 'rShoulder':
                    cur_data.append(line[3+cur_a_idx*3+0])
                    cur_data.append(line[3+cur_a_idx*3+1])
                    cur_data.append(line[3+cur_a_idx*3+2]+45)
                else:
                    cur_data.append(line[3+cur_a_idx*3+0])
                    cur_data.append(line[3+cur_a_idx*3+1])
                    cur_data.append(line[3+cur_a_idx*3+2])
        motion_data.append(cur_data)
    motion_data = np.asarray(motion_data)
    return motion_data
