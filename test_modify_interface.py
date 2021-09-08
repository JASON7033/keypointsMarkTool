import os
import json
import numpy as np
import cv2
import copy
import time


def scenetype_img2bboxs(json_base, trackName='tracks_joints.json'):
    track_json = os.path.join(json_base, trackName)
    seqinfo_json = os.path.join(json_base, 'seqinfo.json')
    with open(track_json, 'r') as rf:
        track_data = json.load(rf)
    with open(seqinfo_json, 'r') as rf:
        seqinfo = json.load(rf)
    seq_list = seqinfo['imUrls']
    img_width, img_height = seqinfo['imWidth'], seqinfo['imHeight']
    scenetype_img2data_dict = {}
    sceneName2frameId_dict = {}
    total_bbox_num = 0
    final_bbox_num = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        frames = track_data[i]['frames']
        for person_dict in frames:
            # print('%d - %d' %(track_id, person_dict['frame id']))
            scene_name = seq_list[person_dict['frame id'] - 1]
            sceneName2frameId_dict[scene_name] = person_dict['frame id']
            total_bbox_num += 1
            # filter disappear
            occlusion_state = person_dict['occlusion']
            if occlusion_state == 'disappear':
                pass  # Panda dataset occlusion is not correct, so retain disappear.
                # continue
            if 'state' in person_dict and person_dict['state'] == 'delete bbox':
                continue
            final_bbox_num += 1
            if scene_name not in scenetype_img2data_dict:
                scenetype_img2data_dict[scene_name] = {}
                scenetype_img2data_dict[scene_name]['bboxs_list'] = []
                scenetype_img2data_dict[scene_name]['trackId_list'] = []
                scenetype_img2data_dict[scene_name]['kps_list'] = []
            rect = person_dict['rect']
            w1 = int(rect['tl']['x'] * img_width)
            h1 = int(rect['tl']['y'] * img_height)
            w2 = int(rect['br']['x'] * img_width)
            h2 = int(rect['br']['y'] * img_height)

            if 'coco keypoints' in person_dict:
                kps_list = person_dict['coco keypoints']
                cur_kps = np.array(kps_list).reshape(-1, 3)
                assert cur_kps.shape[0] == 17, 'keypoints are not illegal'
                cur_kps[:, 0] = cur_kps[:, 0] * img_width
                cur_kps[:, 1] = cur_kps[:, 1] * img_height
            else:
                cur_kps = None

            scenetype_img2data_dict[scene_name]['bboxs_list'].append((w1, h1, w2, h2))
            scenetype_img2data_dict[scene_name]['trackId_list'].append(track_id)
            scenetype_img2data_dict[scene_name]['kps_list'].append(cur_kps)

    # print('%s  ,after appear filter, %d -> %d' % (seqinfo['name'], total_bbox_num, final_bbox_num))

    return scenetype_img2data_dict, track_data, seqinfo, sceneName2frameId_dict, [img_width, img_height]


def visual_bbox(scene_image, bbox_list, trackId_list, only_show_by_trackId=[]):
    # if only_show_by_trackId=[5], only show bbox which trackId==5
    for i, bbox in enumerate(bbox_list):
        if len(only_show_by_trackId) != 0 and trackId_list[i] not in only_show_by_trackId:
            continue
        color = np.array(np.random.rand(3)) * 255
        rec_thickness = max(int((bbox[3] - bbox[1]) * 0.01), 2)
        text_size = max(int((bbox[3] - bbox[1]) * 0.005), 2)
        cv2.rectangle(scene_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color,  # bbox: w1, h1, w2, h2
                      rec_thickness)
        cv2.putText(scene_image, str(trackId_list[i]), (bbox[0], bbox[1] - rec_thickness), cv2.FONT_HERSHEY_PLAIN,
                    text_size, color,
                    thickness=text_size)
    return scene_image


def visual_bbox_single(image, bbox, trackId):
    color = np.array(np.random.rand(3)) * 255
    rec_thickness = max(int((bbox[3] - bbox[1]) * 0.01), 2)
    text_size = max(int((bbox[3] - bbox[1]) * 0.005), 2)
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color,  # bbox: w1, h1, w2, h2
                  rec_thickness)
    cv2.putText(image, str(trackId), (bbox[0], bbox[1] - rec_thickness), cv2.FONT_HERSHEY_PLAIN,
                text_size, color,
                thickness=text_size)
    return image


def visual_kps(scene_image, kps_list, trackId_list, only_show_by_trackId=[]):
    # if only_show_by_trackId=[5], only show bbox which trackId==5
    for i, kps in enumerate(kps_list):
        if len(only_show_by_trackId) != 0 and trackId_list[i] not in only_show_by_trackId:
            continue
        if kps is None:
            continue
        color = np.array(np.random.rand(3)) * 255
        per_kps = kps[:, :2]
        circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.1)
        for coord in per_kps:
            x_coord, y_coord = int(coord[0]), int(coord[1])
            cv2.circle(scene_image, (x_coord, y_coord), circle_size, color, -1)
    return scene_image


def visual_kps_limbs(scene_image, kps_list, trackId_list, only_show_by_trackId=[], color_type='kps'):
    format = 'coco'
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    for i, kps in enumerate(kps_list):
        if len(only_show_by_trackId) != 0 and trackId_list[i] not in only_show_by_trackId:
            continue
        if kps is None:
            continue

        part_line = {}

        # draw kps
        color = np.array(np.random.rand(3)) * 255
        per_kps = kps[:, :2]
        kp_scores = kps[:, 2]
        circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.05) + 1
        for i, coord in enumerate(per_kps):
            x_coord, y_coord = int(coord[0]), int(coord[1])
            part_line[i] = (x_coord, y_coord)
            if color_type == 'kps':
                color = p_color[i]
            cv2.circle(scene_image, (x_coord, y_coord), circle_size, color, -1)

        # draw limb
        limb_size = circle_size
        for i, (start_p, end_p) in enumerate(l_pair):
            if start_p in part_line and end_p in part_line:
                start_xy = part_line[start_p]
                end_xy = part_line[end_p]
                if color_type == 'kps':
                    color = p_color[i]
                if i < len(line_color):
                    cv2.line(scene_image, start_xy, end_xy, color, limb_size)

    return scene_image


def visual_kps_limbs_single(image, kps, color_type='kps'):
    format = 'coco'
    if format == 'coco':
        l_pair = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
            (17, 11), (17, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]
        p_color = [(0, 255, 255), (0, 191, 255), (0, 255, 102), (0, 77, 255), (0, 255, 0),
                   # Nose, LEye, REye, LEar, REar
                   (77, 255, 255), (77, 255, 204), (77, 204, 255), (191, 255, 77), (77, 191, 255), (191, 255, 77),
                   # LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
                   (204, 77, 255), (77, 255, 204), (191, 77, 255), (77, 255, 191), (127, 77, 255), (77, 255, 127),
                   (0, 255, 255)]  # LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
        line_color = [(0, 215, 255), (0, 255, 204), (0, 134, 255), (0, 255, 50),
                      (77, 255, 222), (77, 196, 255), (77, 135, 255), (191, 255, 77), (77, 255, 77),
                      (77, 222, 255), (255, 156, 127),
                      (0, 127, 255), (255, 127, 77), (0, 77, 255), (255, 77, 36)]

    part_line = {}

    # draw kps
    color = np.array(np.random.rand(3)) * 255
    per_kps = kps[:, :2]
    kp_scores = kps[:, 2]
    circle_size = int(np.sqrt(np.sum((per_kps[5] - per_kps[12]) ** 2)) * 0.05) + 1
    for i, coord in enumerate(per_kps):
        x_coord, y_coord = int(coord[0]), int(coord[1])
        part_line[i] = (x_coord, y_coord)
        if color_type == 'kps':
            color = p_color[i]
        cv2.circle(image, (x_coord, y_coord), circle_size, color, -1)

    # draw limb
    limb_size = circle_size
    for i, (start_p, end_p) in enumerate(l_pair):
        if start_p in part_line and end_p in part_line:
            start_xy = part_line[start_p]
            end_xy = part_line[end_p]
            if color_type == 'kps':
                color = p_color[i]
            if i < len(line_color):
                cv2.line(image, start_xy, end_xy, color, limb_size)

    return image


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_anno_json(track_data, json_name):
    with open(json_name, 'w') as f_obj:
        json.dump(track_data, f_obj)


def delete_bbox(track_data, trackId, frameId):
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        frames = track_data[i]['frames']
        for j, person_dict in enumerate(frames):
            if frameId != person_dict['frame id']:
                continue
            track_data[i]['frames'][j]['state'] = 'delete bbox'
            print('delete bbox, track_id = %d, frameId = %d' % (trackId, frameId))
            return track_data
    print('not find the expected delete track_id = %d, frameId = %d' % (trackId, frameId))
    return


def recovery_delete_bbox(track_data, trackId, frameId):
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        frames = track_data[i]['frames']
        for j, person_dict in enumerate(frames):
            if frameId != person_dict['frame id']:
                continue
            if 'state' in track_data[i]['frames'][j] and 'state' in track_data[i]['frames'][j][
                'state'] == 'delete bbox':
                track_data[i]['frames'][j].pop('state')
                print('recovery bbox, track_id = %d, frameId = %d' % (trackId, frameId))
            else:
                print('not need to recovery bbox, track_id = %d, frameId = %d' % (trackId, frameId))
            return track_data
    print('not find the expected recovery track_id = %d, frameId = %d' % (trackId, frameId))
    return


def delete_bbox_frame(track_data, trackId, frameId):  # for making a mistake to add a error bbox with existing trackId
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        frames = track_data[i]['frames']
        if len(frames) == 1:
            print('the lens fo frames shouble be greater than 1')
            return
        for j, person_dict in enumerate(frames):
            if frameId != person_dict['frame id']:
                continue
            if 'state' not in track_data[i]['frames'][j] or track_data[i]['frames'][j]['state'] != 'add bbox':
                print('this delete is not legal')
                return
            track_data[i]['frames'] = track_data[i]['frames'][:j] + track_data[i]['frames'][j + 1:]
            print('delete frame, track_id = %d, frameId = %d' % (trackId, frameId))
            return track_data
    print('not find the expected delete track_id = %d, frameId = %d' % (trackId, frameId))
    return


def delete_bbox_trackId(track_data, trackId):
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        track_data = track_data[:i] + track_data[i + 1:]
        print('delete trackId', trackId)
        return track_data
    print('not find the expected delete trackId', trackId)
    return


def getNew_trackId(track_data):
    max_trackId = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id > max_trackId:
            max_trackId = track_id
    newId = max_trackId + 1
    print('generate new trackId', newId)
    return newId


def get_max_trackId(track_data):
    max_trackId = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id > max_trackId:
            max_trackId = track_id
    print('current max trackId %d' % max_trackId)
    return max_trackId


def add_new_bbox(track_data, trackId, frameId, bbox, scene_shape, face_origentation='', occlusion=''):
    if not isBbox(bbox):
        print('bbox is not legal')
        return
    if face_origentation not in ['', "back", "front", "left", "left back", "left front", "right", "right back",
                                 "right front", "unsure"]:
        print('face origentation input is wrong')
        return
    if occlusion not in ['', "normal", "hide", "serious hide", "disappear"]:
        print('occlusion input is wrong')
        return
    scene_width, scene_height = scene_shape[0], scene_shape[1]
    w1, h1, w2, h2 = bbox[0], bbox[1], bbox[2], bbox[3]
    new_rect_dict = {"tl": {}, "br": {}}
    new_rect_dict['tl']['x'] = w1 / scene_width
    new_rect_dict['tl']['y'] = h1 / scene_height
    new_rect_dict['br']['x'] = w2 / scene_width
    new_rect_dict['br']['y'] = h2 / scene_height
    new_frame_dict = {'frame id': int(frameId)}
    new_frame_dict['rect'] = new_rect_dict
    new_frame_dict['face orientation'] = face_origentation
    new_frame_dict['occlusion'] = occlusion
    new_frame_dict['state'] = 'add bbox'

    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        # find exist trackId, add existing Id bbox
        frames = track_data[i]['frames']
        insert_index = 0
        for j in range(len(frames)):  # frameId is sorted
            cur_frame_id = frames[j]['frame id']
            if cur_frame_id > frameId:
                break
            if cur_frame_id == frameId:
                print('add bbox trackId %d frameId %d has existed' % (trackId, frameId))
                return track_data
            insert_index += 1

        track_data[i]['frames'] = track_data[i]['frames'][:insert_index] + [new_frame_dict] + track_data[i]['frames'][
                                                                                              insert_index:]
        print('add a new bbox with existed trackId %d' % trackId)
        return track_data

    # not find, add newId bbox
    new_track_dict = {'track id': int(trackId)}
    new_track_dict['frames'] = [new_frame_dict]
    track_data.append(new_track_dict)
    print('add a new bbox with new trackId %d' % trackId)
    return track_data


def delete_bbox_real(track_data, trackId, frameId):
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue
        frames = track_data[i]['frames']
        if len(frames) == 1:
            track_data = track_data[:i] + track_data[i + 1:]
            print('delete trackId', trackId)
            return track_data

        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            if cur_frame_id == frameId:
                print('delete bbox frame, trackId %d frameId %d' % (trackId, frameId))
                track_data[i]['frames'] = track_data[i]['frames'][:j] + track_data[i]['frames'][j + 1:]
                return track_data
        print('not find the real delete bbox with frame %d' % frameId)

    # not find
    print('not find the real delete bbox with trackId %d' % trackId)
    return track_data


def adjust_bbox(track_data, trackId, frameId, bbox, scene_shape):
    scene_width, scene_height = scene_shape[0], scene_shape[1]

    w1, h1, w2, h2 = bbox[0], bbox[1], bbox[2], bbox[3]
    new_rect_dict = {"tl": {}, "br": {}}
    new_rect_dict['tl']['x'] = w1 / scene_width
    new_rect_dict['tl']['y'] = h1 / scene_height
    new_rect_dict['br']['x'] = w2 / scene_width
    new_rect_dict['br']['y'] = h2 / scene_height

    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue

        frames = track_data[i]['frames']
        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            if cur_frame_id == frameId:
                if 'state' in track_data[i]['frames'][j] and track_data[i]['frames'][j]['state'] == 'delete bbox':
                    print('the bbox state is delete bbox, if you want to adjust it, first to recovery it')
                    return
                else:
                    track_data[i]['frames'][j]['state'] = 'adjust bbox'
                    # track_data[i]['frames'][j]['orign_rect'] = track_data[i]['frames'][j]['rect']

                track_data[i]['frames'][j]['rect'] = new_rect_dict
                print('adjust a bbox with existed trackId %d' % trackId)
                return track_data
        print('For trackId %d, there is no frameId %d' % (trackId, frameId))
        return
    print('There is no this trackId', trackId)
    return


def update_json(track_data, frame_id, final_kps, kps_state, scene_shape):  # update track_data
    scene_width, scene_height = scene_shape[0], scene_shape[1]
    final_kps = copy.deepcopy(final_kps)
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id not in kps_state:
            continue
        kps_index = kps_state.index(track_id)
        cur_kps = final_kps[kps_index]
        cur_kps[:, 0] = cur_kps[:, 0] / scene_width
        cur_kps[:, 1] = cur_kps[:, 1] / scene_height
        cur_kps_list = list(cur_kps.reshape(-1))
        # print('kps_list len:', len(cur_kps_list))

        frames = track_data[i]['frames']
        for j, person_dict in enumerate(frames):
            if frame_id != person_dict['frame id']:
                continue
            track_data[i]['frames'][j]['coco keypoints'] = cur_kps_list
    return track_data


def isBbox(bbox):
    if len(bbox) != 4:
        return False
    w1, h1, w2, h2 = bbox
    if w1 < w2 and h1 < h2:
        return True
    else:
        return False


def isKps(kps):
    ###
    return True


def add_or_replace_kps(track_data, trackId, frameId, kps, scene_shape):
    if not isKps(kps):
        print('kps is not legal')
        return
    scene_width, scene_height = scene_shape[0], scene_shape[1]

    kps[:, 0] = kps[:, 0] / scene_width
    kps[:, 1] = kps[:, 1] / scene_height
    kps_list = list(kps.reshape(-1))

    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if track_id != trackId:
            continue

        frames = track_data[i]['frames']
        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            if cur_frame_id == frameId:
                if 'state' in track_data[i]['frames'][j] and track_data[i]['frames'][j]['state'] == 'delete bbox':
                    print('the bbox state is delete bbox, if you want to add kps for it, first to recovery it')
                    return
                else:
                    if 'coco keypoints' in track_data[i]['frames'][j]:
                        print('replace kps for trackId %d, frameId %d' % (trackId, frameId))
                    else:
                        print('add new kps for trackId %d, frameId %d' % (trackId, frameId))
                    track_data[i]['frames'][j]['coco keypoints'] = kps_list
                    return track_data
        print('For trackId %d, there is no frameId %d' % (trackId, frameId))
        return
    print('There is no this trackId', trackId)
    return


def parseFromLabelMe(jsonName):
    with open(jsonName, 'r') as rf:
        data = json.load(rf)
    shapes_list = data['shapes']
    if len(shapes_list) < 1:
        print('parse add 0 bbox')
        return
    add_list = []
    adjust_list = []
    parseDict = {}

    for shape_dict in shapes_list:
        if shape_dict['shape_type'] != 'rectangle':
            continue
        w1 = shape_dict['points'][0][0]
        h1 = shape_dict['points'][0][1]
        w2 = shape_dict['points'][1][0]
        h2 = shape_dict['points'][1][1]
        trackId = int(shape_dict['group_id'])
        if shape_dict['label'] == 'add':
            # add a bbox
            add_list.append([trackId, (w1, h1, w2, h2)])
        elif shape_dict['label'] == 'adjust':
            # adjust a bbox
            adjust_list.append([trackId, (w1, h1, w2, h2)])
    parseDict['add_list'] = add_list
    parseDict['adjust_list'] = adjust_list
    return parseDict


def add_or_adjust_bbox_from_json(jsonName, track_data, frameId, scene_shape):
    parseDict = parseFromLabelMe(jsonName)
    add_list = parseDict['add_list']
    adjust_list = parseDict['adjust_list']

    # add new bbox
    for i in range(len(add_list)):
        trackId = add_list[i][0]
        bbox = add_list[i][1]
        bbox = [int(x) for x in bbox]
        track_data = add_new_bbox(track_data, trackId, frameId, bbox, scene_shape, face_origentation='', occlusion='')

    # adjust bbox
    for i in range(len(adjust_list)):
        trackId = adjust_list[i][0]
        bbox = adjust_list[i][1]
        bbox = [int(x) for x in bbox]
        track_data = adjust_bbox(track_data, trackId, frameId, bbox, scene_shape)
    return track_data


def generate_kps_crop(scene_image, track_data, frameId, crop_path, extra_trackId_list=[]):
    img_height, img_width = scene_image.shape[:2]
    crop_count = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        frames = track_data[i]['frames']
        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            person_dict = frames[j]
            if cur_frame_id == frameId:
                if 'state' in person_dict and person_dict['state'] == 'delete bbox':
                    continue
                if person_dict['occlusion'] == 'disappear':
                    continue
                if 'coco keypoints' in track_data[i]['frames'][j] and track_id not in extra_trackId_list:
                    continue
                else:
                    # crop
                    rect = person_dict['rect']
                    w1 = max([int(rect['tl']['x'] * img_width), 0])
                    h1 = max([int(rect['tl']['y'] * img_height), 0])
                    w2 = min([int(rect['br']['x'] * img_width), img_width])
                    h2 = min([int(rect['br']['y'] * img_height), img_height])
                    crop = scene_image[h1:h2, w1:w2, :]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        crop_name = str(track_id) + '_' + str(w1) + '_' + str(h1) + '_' + str(w2) + '_' + str(
                            h2) + '.jpg'
                        cv2.imwrite(os.path.join(crop_path, crop_name), crop)
                        crop_count += 1

    print('frameId %d , generate %d kps crops' % (frameId, crop_count))
    return

def generate_kps_crop2(scene_image, track_data, frameId, crop_path, extra_path, extra_trackId_list=[]):
    img_height, img_width = scene_image.shape[:2]
    crop_count = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        frames = track_data[i]['frames']
        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            person_dict = frames[j]
            if cur_frame_id == frameId:
                if 'state' in person_dict and person_dict['state'] == 'delete bbox':
                    continue
                if person_dict['occlusion'] == 'disappear':
                    continue
                if 'coco keypoints' in track_data[i]['frames'][j] and track_id not in extra_trackId_list:
                    continue
                else:
                    # crop
                    rect = person_dict['rect']
                    w1 = max([int(rect['tl']['x'] * img_width), 0])
                    h1 = max([int(rect['tl']['y'] * img_height), 0])
                    w2 = min([int(rect['br']['x'] * img_width), img_width])
                    h2 = min([int(rect['br']['y'] * img_height), img_height])
                    crop = scene_image[h1:h2, w1:w2, :]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        crop_name = str(track_id) + '_' + str(w1) + '_' + str(h1) + '_' + str(w2) + '_' + str(
                            h2) + '.jpg'
                        if track_id in extra_trackId_list:
                            cv2.imwrite(os.path.join(extra_path, crop_name), crop)
                        else:
                            cv2.imwrite(os.path.join(crop_path, crop_name), crop)
                        crop_count += 1

    print('frameId %d , generate %d kps crops' % (frameId, crop_count))
    return

def generate_check_legal_kps_crop(scene_image, track_data, frameId, crop_path, tradId_list=[]):
    img_height, img_width = scene_image.shape[:2]
    crop_count = 0
    for i in range(len(track_data)):
        track_id = track_data[i]['track id']
        if len(tradId_list) > 0 and track_id not in tradId_list:
            continue
        frames = track_data[i]['frames']
        for j in range(len(frames)):
            cur_frame_id = frames[j]['frame id']
            person_dict = frames[j]
            if cur_frame_id == frameId:
                if 'state' in person_dict and person_dict['state'] == 'delete bbox':
                    continue
                if person_dict['occlusion'] == 'disappear':
                    continue

                # crop
                rect = person_dict['rect']
                w1 = max([int(rect['tl']['x'] * img_width), 0])
                h1 = max([int(rect['tl']['y'] * img_height), 0])
                w2 = min([int(rect['br']['x'] * img_width), img_width])
                h2 = min([int(rect['br']['y'] * img_height), img_height])
                bbox = [w1, h1, w2, h2]

                radio = 0.5
                w = w2 - w1
                h = h2 - h1
                w1_crop = max(0, int(w1 - w * radio))
                h1_crop = max(0, int(h1 - h * radio))
                w2_crop = min(img_width, int(w2 + w * radio))
                h2_crop = min(img_height, int(h2 + h * radio))

                crop = scene_image[h1_crop:h2_crop, w1_crop:w2_crop, :]
                crop = copy.deepcopy(crop)
                if w1 > 0 and w2 > 0 and h1 > 0 and h2 > 0 and w1 < w2 and h1 < h2:
                    crop_name = str(track_id) + '_' + str(w1_crop) + '_' + str(h1_crop) + '_' + str(
                        w2_crop) + '_' + str(
                        h2_crop) + '.jpg'
                    bbox[0] = bbox[0] - w1_crop
                    bbox[1] = bbox[1] - h1_crop
                    bbox[2] = bbox[2] - w1_crop
                    bbox[3] = bbox[3] - h1_crop
                    visual_bbox_single(crop, bbox, track_id)
                    if 'coco keypoints' in track_data[i]['frames'][j]:
                        kps_list = person_dict['coco keypoints']
                        cur_kps = np.array(kps_list).reshape(-1, 3)
                        assert cur_kps.shape[0] == 17, 'keypoints are not illegal'
                        cur_kps[:, 0] = cur_kps[:, 0] * img_width
                        cur_kps[:, 1] = cur_kps[:, 1] * img_height
                        cur_kps[:, 0] -= w1_crop
                        cur_kps[:, 1] -= h1_crop
                        visual_kps_limbs_single(crop, cur_kps, color_type='kps')
                    cv2.imwrite(os.path.join(crop_path, crop_name), crop)
                    crop_count += 1

    return


def read_anno_from_txt(fileName):
    confident = 0.9
    kps = None
    with open(fileName, 'r') as rf:
        data = rf.readlines()
    if len(data) == 18:
        print("fileName_of_txt (in processing): "+fileName)
        # print(fileName.split('\\'))
        # temp = fileName.split('\\')[-1]
        temp = fileName.split('/')[-1]
        temp = temp.split('_')
        print(temp)
        trackId = int(temp[0])
        w1 = int(temp[1])
        h1 = int(temp[2])
        w2 = int(temp[3])
        h2 = int(temp[4].replace('.txt', ''))
        w = w2 - w1
        h = h2 - h1
        kps = np.ones((17, 3)) * confident
        for i in range(len(data) - 1):
            num_list = data[i + 1].split()
            kps[i, 0] = int(float(num_list[0]) * w + w1)
            kps[i, 1] = int(float(num_list[1]) * h + h1)
        return (trackId, kps)
    return None


def add_or_replace_kps_from_crop_anno(track_data, frameId, crop_path, scene_shape):
    # file_list = sorted(os.listdir(crop_path))
    file_list=sorted(os.listdir(crop_path))
    for i in range(len(file_list)):
        fileName=os.path.join(crop_path,file_list[i])
        print("this txt filename is :" + fileName)
        res=read_anno_from_txt(fileName)
        print("read_anno_file_txt:"+fileName)
        if res is not None:
            trackId,kps=res
            add_or_replace_kps(track_data,trackId,frameId,kps,scene_shape)
    return track_data
    # for i in range(len(file_list)):
    #     fileName = os.path.join(crop_path, file_list[i])
    #     res = read_anno_from_txt(fileName)
    #     print("read_anno_from_txt : "+fileName)
    #     if res is not None:
    #         trackId, kps = res
    #         add_or_replace_kps(track_data, trackId, frameId, kps, scene_shape)
    # return track_data


if __name__ == '__main__':
    anno_base = 'panda_video_merge_pose_result'
    # anno_base='/Users/jason/Desktop/标注任务/anno/10_Huaqiangbei'
    # orign_scene_base = 'F:/wh_backup/wh_workspace/Panda/panda_video/train'
    orign_scene_base = '/Users/jason/Desktop/makedata/panda_video_merge_pose_result/train'
    # orgin_scene_base='/Users/jason/Desktop/makedata/panda_video_merge_pose_result/10_Huaqiangbei'
    image_check_base = 'image_check'

    scene_type_list = sorted(os.listdir(anno_base))
    # print("*"*10)
    # print(scene_type_list)
    scene_type = scene_type_list[-2]  #adjust scene of 10
    json_base = os.path.join(anno_base, scene_type)
    print("*"*10)
    print(json_base)
    trackName = 'tracks_joints_modify.json'
    scenetype_img2data_dict, track_data, seqinfo, sceneName2frameId_dict, scene_shape = scenetype_img2bboxs(
        json_base, trackName=trackName)
    print("*"*10)
    print(seqinfo) #infos of this scene
    scene_name_list = seqinfo["imUrls"]  # has been sorted
    check_image_save_path = os.path.join(image_check_base, scene_type)
    makedir(check_image_save_path)


    # for step in range(-117,-11):
    for step in range(-115,0):
        scene_name = scene_name_list[step]   #each pic changed here
        print("*"*10)
        print(scene_name)

        modify_flag = 'add kps'
        visual_flag = ['bbox', 'kps']
        if modify_flag in ['del bbox', 'recovery bbox', 'add bbox', 'kps crop', 'real del bbox', 'add kps', 'check crop']:
            print(modify_flag)
            frameId = sceneName2frameId_dict[scene_name]

            # ***generate kps crop for modify and blank separately***
            if modify_flag == 'kps crop':
                # extra_trackId_list = [498, 512, 513]    标注需要修改的track id的编号
                extra_trackId_list = []
                extra_file_path=os.path.join("image_check/10_Huaqiangbei/check_crops",scene_name.split('.')[0],'final.txt')
                print(extra_file_path)
                with open(extra_file_path,'r') as f:
                    contents=f.read()
                    content= contents.replace('\n','')
                    trackIds=content.split(',')[0:-1] #取到-1因为最后一行有一个\n->''会导致程序报错
                    for trackId in trackIds:
                        extra_trackId_list.append(int(trackId))
                print("extra_trackId_list")
                print(extra_trackId_list)

                scene_image = cv2.imread(os.path.join(orign_scene_base, scene_type, scene_name))
                print(os.path.join(orign_scene_base, scene_type, scene_name))
                crop_path = os.path.join(check_image_save_path, 'select_crops', scene_name.replace('.jpg', ''))
                extra_path = os.path.join(check_image_save_path, 'modify_crops', scene_name.replace('.jpg', ''))
                kps_anno_path = os.path.join(check_image_save_path, 'annos', scene_name.replace('.jpg', ''))
                makedir(crop_path)
                makedir(extra_path)
                makedir(kps_anno_path)
                generate_kps_crop2(scene_image, track_data, frameId, crop_path, extra_path,
                                   extra_trackId_list=extra_trackId_list)
                track_data = None

            # ***delete bboxs (Pseudo deletion)***
            if modify_flag == 'del bbox':
                delete_trackId_list = [64, 65]
                for delete_trackId in delete_trackId_list:
                    track_data = delete_bbox(track_data, delete_trackId, frameId)  # delete a bbox
            # ***recovery bboxs (only recovery delete by del bbox)***
            if modify_flag == 'recovery bbox':
                delete_trackId_list = [64, 65]
                for delete_trackId in delete_trackId_list:
                    track_data = recovery_delete_bbox(track_data, delete_trackId, frameId)  # recover the bbox

            # ***add bbox from json***
            if modify_flag == 'add bbox':
                print(check_image_save_path)
                add_bbox_json_name = os.path.join(check_image_save_path, scene_name.replace('.jpg', '_check_modify.json'))
                track_data = add_or_adjust_bbox_from_json(add_bbox_json_name, track_data, frameId, scene_shape)

            # ***Not advise, real delete bbox***
            if modify_flag == 'real del bbox':
                delete_trackId_list = [296]
                for delete_trackId in delete_trackId_list:
                    track_data = delete_bbox_real(track_data, delete_trackId, frameId)



            # ***generate kps crop***
            if modify_flag == 'kps crop backup':
                extra_trackId_list = [1]
                scene_image = cv2.imread(os.path.join(orign_scene_base, scene_type, scene_name))
                crop_path = os.path.join(check_image_save_path, 'crops', scene_name.replace('.jpg', ''))
                kps_anno_path = os.path.join(check_image_save_path, 'annos', scene_name.replace('.jpg', ''))
                makedir(crop_path)
                makedir(kps_anno_path)
                generate_kps_crop(scene_image, track_data, frameId, crop_path, extra_trackId_list=extra_trackId_list)
                track_data = None


            # ***add kps from crop anno txt***
            if modify_flag == 'add kps':
                print("chekc_image_save_path:"+check_image_save_path)
                kps_anno_path = os.path.join(check_image_save_path, 'annos', scene_name.replace('.jpg', ''))
                track_data = add_or_replace_kps_from_crop_anno(track_data, frameId, kps_anno_path, scene_shape)

            if modify_flag == 'check crop':
                scene_image = cv2.imread(os.path.join(orign_scene_base, scene_type, scene_name))
                crop_path = os.path.join(check_image_save_path, 'check_crops', scene_name.replace('.jpg', ''))
                makedir(crop_path)
                generate_check_legal_kps_crop(scene_image, track_data, frameId, crop_path, tradId_list=[])
                track_data = None
            # ***add a new bbox with new trackId***
            # new_trackId=getNew_trackId(track_data)
            # new_bbox=(10000, 10000, 12500, 11000) # w1 h1 w2 h2
            # new_trackId=296
            # track_data=add_new_bbox(track_data, new_trackId, frameId, new_bbox, scene_shape, face_origentation='', occlusion='')
            # print(new_trackId) # 296
            # ***delete the trackId
            # track_data=delete_bbox_trackId(track_data, 296)

            # ***add a new bbox with existed trackId***  scene_name = scene_name_list[3] trackId 1
            # oldId = 1  # exist in scene_name_list[0:3]
            # track_data = add_new_bbox(track_data, oldId, frameId, (10000, 10000, 12500, 11000), scene_shape,
            #                           face_origentation='',
            #                           occlusion='')
            # track_data = delete_bbox_frame(track_data, 1, frameId)

            # ***adjust bbox***
            # track_data=adjust_bbox(track_data, 50, frameId, (10000, 10000, 12500, 11000), scene_shape)

            if track_data is not None:
                # backup
                back_up_save_path = os.path.join(check_image_save_path, 'back_up',
                                                 time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
                makedir(back_up_save_path)
                copy_command = 'cp ' + os.path.join(json_base, trackName) + ' ' + os.path.join(back_up_save_path,
                                                                                                 trackName)
                # print(copy_command)
                os.system(copy_command)
                save_anno_json(track_data, os.path.join(json_base, trackName))

        if len(visual_flag) > 0 and ('bbox' in visual_flag or 'kps' in visual_flag):
            scenetype_img2data_dict, track_data, seqinfo, sceneName2frameId_dict, scene_shape = scenetype_img2bboxs(
                json_base, trackName='tracks_joints_modify.json')
            cur_bboxs_list = scenetype_img2data_dict[scene_name]['bboxs_list']
            cur_trackId_list = scenetype_img2data_dict[scene_name]['trackId_list']
            cur_kps_list = scenetype_img2data_dict[scene_name]['kps_list']
            scene_image = cv2.imread(os.path.join(orign_scene_base, scene_type, scene_name))
            if 'bbox' in visual_flag:
                scene_image = visual_bbox(scene_image, cur_bboxs_list, cur_trackId_list, only_show_by_trackId=[])
            if 'kps' in visual_flag:
                scene_image = visual_kps_limbs(scene_image, cur_kps_list, cur_trackId_list, only_show_by_trackId=[],
                                               color_type='kps')

            image_save_path = os.path.join(check_image_save_path, scene_name.replace('.jpg', '_check_modify.jpg'))
            cv2.imwrite(image_save_path, scene_image)
            print('visul', visual_flag, image_save_path)
            max_trackId = get_max_trackId(track_data)
