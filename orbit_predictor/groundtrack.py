class BaseElevationAPI:
    def get_elevation(self, *, longitude, latitude):
        raise NotImplementedError

    def get_ground_point(self, position):
        latitude, longitude, _ = position.position_llh
        return (
            longitude,
            latitude,
            self.get_elevation(longitude=longitude, latitude=latitude),
        )

    def get_groundtrack(self, positions):
        return [self.get_ground_point(position) for position in positions]


class ZeroElevation(BaseElevationAPI):
    def get_elevation(self, *, longitude, latitude):
        return 0.0


def compute_groundtrack(predictor, times, elevation_api=ZeroElevation()):
    positions = [predictor.get_position(time) for time in times]
    return elevation_api.get_groundtrack(positions)
