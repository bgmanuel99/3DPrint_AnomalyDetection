import cv2
import numpy as np
from typing import List
from imutils import contours
from imutils import perspective
from scipy.spatial import distance as dist

from app.common.common_functionalities import CommonFunctionalities
from app.common.common_morphology_operations import CommonMorphologyOperations
from app.common.common_prints import CommonPrints

class ImageSegmetation(object):
    """This class contains methods to segmentated the original image with the
    3D printed and reference objects and to get a pixels per metric value 
    based on the reference object

    Methods:
        segment_image (image: np.ndarray): 
            Method to obtain the 3d printed object mask and different data to
            calculate the perfect model and areas
        _3d_printed_object_process (
                image: np.ndarray, 
                cnt: np.ndarray, 
                segmented: np.ndarray):
            Private method to obtain the masked 3d printed object and all the 
            needed data to calculate the perfect models
        _reference_object_process (
                image: np.ndarray, 
                cnt: np.ndarray, 
                segmented: np.ndarray):
            Private method to obtain the reference object area
        _get_complete_segmented_image (image: np.ndarray): 
            Private method to obtain the complete segmentation of the original 
            image
        _get_contours (segmented: np.ndarray): 
            Private method to obtain the contours of the objects of the 
            original image
        _get_masked_object_by_contour (
                image: np.ndarray, 
                cnt: np.ndarray, 
                segmented: np.ndarray):
            Private method to mask an object of an image based on its contour
        _get_data_from_box_coordinates (
                box_coordinates: tuple[tuple[float]]):
            Private method to obtain all neccesary data from the box 
            coordinates of a contour needed for translation purposes in an 
            image
        _transform_and_translate_masked_object (
                masked_object: np.ndarray, 
                top_left_coord: tuple[float], 
                box_coordinates: tuple[tuple[float]], 
                use_external_contour: bool):
            Private method to get the perspective of an object in an image
        _mid_point (
                point_A: tuple[float], 
                point_B: tuple[float]):
            Private method to calculate the mid points of an edge
    """
    
    @classmethod
    def segment_image(cls, image: np.ndarray) -> tuple[
            np.ndarray, List[float], tuple[float], tuple[float], float, float]:
        """Method to obtain the 3d printed object mask and different data to
        calculate the perfect model and areas

        Parameters:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Segmented image
            List[float]: 
                List of pixels per metric values variations representing degree
                offsets when taking the picture of the original image
            tuple[float]: Middle coordinates of the 3d printed object
            tuple[float]: Top left coordinates of the 3d printed object
            float: Reference object area in pixels
            float: 
                SSIM max score between the images of the segmented reference 
                object and its perfect models
        """
        
        print("[INFO] Segmenting images")
        
        # Segment the original image
        segmented: np.ndarray = cls._get_complete_segmented_image(image)
        
        CommonPrints.print_image("segmented", segmented, 600, True)
        
        # Obtain the contours of the object in the image
        cnts = cls._get_contours(segmented)
        
        contours_image = np.zeros(image.shape, dtype=np.uint8)
        for c in cnts:
            cv2.drawContours(contours_image, [c], -1, (255, 255, 255), 1)
        CommonPrints.print_image("contours_image", contours_image, 600, True)
        
        # 3D printed object process
        (translated_3d_object, 
         middle_coords_3d_object, 
         top_left_coord_3d_object) = cls._3d_printed_object_process(
            image, cnts[1], segmented)
        
        CommonPrints.print_image(
            "translated_3d_object", translated_3d_object, 600, True)
        
        # Reference object process
        (reference_object_pixels_area, 
         ssim_max_score_reference_object, 
         ppm_degree_offset) = cls._reference_object_process(
            image, cnts[0], segmented)
                
        return (translated_3d_object, 
                ppm_degree_offset, 
                middle_coords_3d_object, 
                top_left_coord_3d_object, 
                reference_object_pixels_area, 
                ssim_max_score_reference_object)
        
    @classmethod
    def _3d_printed_object_process(
            cls, 
            image: np.ndarray, 
            cnt: np.ndarray, 
            segmented: np.ndarray) -> tuple[
                np.ndarray, tuple[float], tuple[float]]:
        """Method to obtain the masked 3d printed object and all the needed 
        data to calculate the perfect models

        Parameters:
            image (np.ndarray): 
                Original image with 3d printed and reference objects
            cnt (np.ndarray): Contour of the 3d printed object
            segmented (np.ndarray): 
                Segmentation of the original image

        Returns:
            tuple[np.ndarray, tuple[float], tuple[float]]: 
                - Transformed and translated image with the masked 3d printed 
                object
                - Middle coordinates of the 3d printed object
                - Top left coordinates of the 3d printed object
        """
        
        # Obtain only the masked 3d printed object
        masked_3d_object = cls._get_masked_object_by_contour(
            image, cnt, segmented)
        
        # Calculate 3d printed object box coordinates
        printed_object_box_coordinates = CommonFunctionalities \
            .get_box_coordinates(cnt)
        
        # Calculate middle and top coordinates of the 3d printed object
        (middle_coords_3d_object, 
         top_left_coord_3d_object) = cls._get_data_from_box_coordinates(
            printed_object_box_coordinates)
        
        # Transform and translate 3d printed object to eliminate part of the 
        # distorsion
        translated_3d_object = cls._transform_and_translate_masked_object(
            masked_3d_object, 
            top_left_coord_3d_object, 
            printed_object_box_coordinates, 
            True)
            
        return (translated_3d_object, 
                middle_coords_3d_object, 
                top_left_coord_3d_object)
        
    @classmethod
    def _reference_object_process(
            cls, 
            image: np.ndarray, 
            cnt: np.ndarray, 
            segmented: np.ndarray) -> tuple[float, float, List[float]]:
        """Method to obtain the reference object area

        Parameters:
            image (np.ndarray): 
                Original image with 3d printed and reference objects
            cnt (np.ndarray): Contour of the reference object
            segmented (np.ndarray): 
                Segmentation of the original image

        Returns:
            tuple[float, float, List[float]]: 
                - Reference object area in pixels
                - SSIM max score between the segmented and transformed 
                reference object image and its perfect models
                - List of pixels per metric values variations representing 
                degree offsets when taking the picture of the original image
        """
        
        # Obtain only the masked reference object
        masked_reference_object = cls._get_masked_object_by_contour(
            image, cnt, segmented)
        
        # Calculate reference object box coordinates
        reference_object_box_coordinates = CommonFunctionalities \
            .get_box_coordinates(cnt)
        
        # Calculate middle and top coordinates of the reference object
        (middle_coords_reference_object, 
         top_left_coord_reference_object) = cls._get_data_from_box_coordinates(
            reference_object_box_coordinates)
        
        # Transform and translate reference object to eliminate part of the 
        # distorsion
        translated_reference_object = cls \
            ._transform_and_translate_masked_object(
                masked_reference_object, 
                top_left_coord_reference_object, 
                reference_object_box_coordinates, 
                False)
        
        # Segment the translated object and obtain its contour
        segmented_reference_object = CommonFunctionalities.get_segmented_image(
            translated_reference_object)
        
        CommonPrints.print_image(
            "segmented_reference_object", 
            segmented_reference_object, 
            600)
        
        cnts = CommonFunctionalities \
            .find_and_grab_contours(segmented_reference_object)
        
        cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
        
        # Obtain the contour box coordinates and compute the width of the 
        # object
        box = CommonFunctionalities.get_box_coordinates(cnts[0])
        (top_left, top_right, bottom_right, bottom_left) = perspective \
            .order_points(box)
        
        reference_left_mid_point = cls._mid_point(
            top_left, bottom_left)
        reference_right_mid_point = cls._mid_point(
            top_right, bottom_right)
        
        reference_object_width = dist.euclidean(
            reference_left_mid_point, reference_right_mid_point)
        
        # Obtain a list of pixels per metric values variations representing 
        # degree offsets when taking the picture of the original image
        ppm_degree_offset = []
        
        for offset in [i * 0.1 for i in range(-30, 31)]:
            ppm_degree_offset.append(reference_object_width + offset)
        
        # Obtain reference object perfect models
        referece_object_perfect_models_radius = []
        
        for offset in [i for i in range(-5, 1)]:
            referece_object_perfect_models_radius.append(
                round(reference_object_width/2) + offset)
            
        reference_object_perfect_models = []
        
        for radius in referece_object_perfect_models_radius:
            reference_object_perfect_model = np.zeros(
                shape=segmented.shape, dtype=np.uint8)
            
            reference_object_perfect_models.append(
                cv2.circle(
                    reference_object_perfect_model, 
                    tuple(map(int, middle_coords_reference_object)), 
                    radius, 
                    (255, 255, 255), 
                    -1))
            
        # Calculate best SSIM score between the segmented image of the 
        # reference object in perspective and its perfect models
        ssim_max_score, ssim_max_score_index = CommonFunctionalities \
            .calculate_ssim_max_score(
                segmented_reference_object, reference_object_perfect_models)
        
        # Find perfect model object contour
        cnts = CommonFunctionalities.find_and_grab_contours(
            reference_object_perfect_models[ssim_max_score_index])
        
        # Calculate perfect model object area
        reference_object_pixels_area = cv2.contourArea(cnts[0])
        
        return reference_object_pixels_area, ssim_max_score, ppm_degree_offset
    
    @classmethod
    def _get_complete_segmented_image(cls, image: np.ndarray) -> np.ndarray:
        """Method to obtain the complete segmentation of the original image

        Parameters:
            image (np.ndarray): Original image

        Returns:
            np.ndarray: Complete segmentation of the original image
        """
        
        segmented_image = CommonFunctionalities.get_segmented_image(image)
        
        CommonPrints.print_image("segmented", segmented_image, 600)
        
        opening = CommonMorphologyOperations.morphologyEx_opening(
            segmented_image, (5, 5))
        
        CommonPrints.print_image("opening", opening, 600)
        
        return opening
    
    @classmethod
    def _get_contours(cls, segmented: np.ndarray) -> None:
        """Method to obtain the contours of the objects of the original image

        Parameters:
            segmented (np.ndarray): The segmentation of the original image
        """
        
        # Find contours in the segmented image
        cnts = CommonFunctionalities.find_and_grab_contours(segmented)
        
        # Sort the contours from bottom to top in order to get the contour
        # of the reference object as the first variable
        (cnts, _) = contours.sort_contours(cnts, method="bottom-to-top")
        
        # TODO: Try to do this filter dynamic and remove static 1000
        # Eliminate any contour with an area lesser than a thousand
        return [c for c in cnts if cv2.contourArea(c) > 1000]
        
    @staticmethod
    def _get_masked_object_by_contour(
            image: np.ndarray, 
            cnt: np.ndarray, 
            segmented: np.ndarray):
        """Method to mask an object of an image based on its contour

        Parameters:
            image (np.ndarray): Image with the object
            cnt (np.ndarray): Contour of the object
            segmented (np.ndarray): Segmentation of the passed image

        Returns:
            np.ndarray: Image with the masked object
        """
        
        # Create a mask only for the contour
        filled_contour = np.zeros(image.shape[0:2], dtype=np.uint8)
        
        cv2.fillPoly(filled_contour, [cnt], (255, 255, 255))
        
        # This first bitwise operation is to mask the object of the original 
        # image
        masked = cv2.bitwise_and(image, image, mask=filled_contour)
        
        CommonPrints.print_image("masked", masked, 600)
        
        # This second bitwise operation is to separate the forground from the 
        # background
        masked = cv2.bitwise_and(masked, masked, mask=segmented)
        
        CommonPrints.print_image("masked", masked, 600)
        
        return masked
    
    @classmethod
    def _get_data_from_box_coordinates(
            cls, 
            box_coordinates: tuple[tuple[float]]
            ) -> tuple[tuple[float], tuple[float]]:
        """Method to obtain all neccesary data from the box coordinates of a 
        contour needed for translation purposes in an image
        
        Parameters:
            box_coordinates (tuple[tuple[float]]): 
                The box coordinates of a contour
        
        Returns:
            tuple[float]: Middle coordinates of the box
            tuple[float]: Top left coordinates of the box
        """
        
        (top_left, top_right, bottom_right, bottom_left) = perspective \
            .order_points(box_coordinates)
 
        left_mid_point = cls._mid_point(top_left, bottom_left)
        right_mid_point = cls._mid_point(top_right, bottom_right)
        
        # Compute the middle coordinates of the 3d printed object
        middle_coords = cls._mid_point(left_mid_point, right_mid_point)
        
        return middle_coords, top_left
    
    @classmethod
    def _transform_and_translate_masked_object(
            cls, 
            masked_object: np.ndarray, 
            top_left_coord: tuple[float], 
            box_coordinates: tuple[tuple[float]], 
            use_external_contour: bool) -> np.ndarray:
        """Method to get the perspective of an object in an image

        Parameters:
            masked_object (np.ndarray): 
                Image of the masked object
            top_left_coord (tuple[float]): 
                Top left coordinates of the masked object
            box_coordinates (tuple[tuple[float]]): 
                Box coordinates of the masked object
            use_external_contour:
                Boolean to know whether to use the external contour of the 
                object in the image or the image shape for the translation
                process

        Returns:
            np.ndarray: Object with the perspective transformation
        """
        
        warped = perspective.four_point_transform(
            masked_object, box_coordinates)

        CommonPrints.print_image("warped", warped, 600)
        
        if(use_external_contour):
            segmented_image = CommonFunctionalities.get_segmented_image(warped)
        
            cnts = CommonFunctionalities.find_and_grab_contours(
                segmented_image)
            
            cnts = [c for c in cnts if cv2.contourArea(c) > 1000]
            
            box = CommonFunctionalities.get_box_coordinates(cnts[0])
            (top_left, top_right, bottom_right, bottom_left) = perspective \
                .order_points(box)
            
            translated_object = CommonFunctionalities.get_translated_object(
                warped, 
                top_left,
                top_right, 
                bottom_right, 
                bottom_left, 
                top_left_coord, 
                masked_object.shape, 
                0)
        else:
            translated_object = CommonFunctionalities.get_translated_object(
                warped, 
                (0, 0),
                (warped.shape[1], 0), 
                (warped.shape[1], warped.shape[0]), 
                (0, warped.shape[0]), 
                top_left_coord, 
                masked_object.shape, 
                0)
        
        CommonPrints.print_image("translated_object", translated_object, 600)
        
        return translated_object
    
    @classmethod
    def _mid_point(
            cls, 
            point_A: tuple[float], 
            point_B: tuple[float]) -> tuple[np.float64]:
        """Method to calculate the mid points of an edge

        Parameters:
            point_A (tuple[float]): First set of 2D coordinates
            point_B (tuple[float]): Second set of 2D coordinates

        Returns:
            tuple[np.float64]: Mid points
        """
        
        return (
            (point_A[0] + point_B[0]) * 0.5, 
            (point_A[1] + point_B[1]) * 0.5)