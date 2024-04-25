luv = RGB_to_LUV(self.image['sift'])
        self.image['after_sift'] = luv
        self.display_image(luv, self.sift_la