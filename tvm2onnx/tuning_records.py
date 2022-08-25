import enum
import json
import pathlib
import typing

import parse
from tvm import auto_scheduler, autotvm, meta_schedule

from tvm2onnx.error import InvalidRecordFormatError, TuningRecordTypeError


class RecordType(enum.Enum):
    AUTOSCHEDULE = "autoschedule"
    AUTOTVM = "autotvm"
    METASCHEDULE = "metaschedule"


class TuningRecordFormatError(InvalidRecordFormatError):
    """Indicates that the format of the tuning records was invalid."""


EncodedTuningRecordsType = typing.List[str]
AutoTVMTuningRecordType = typing.List[
    typing.Tuple[autotvm.MeasureInput, autotvm.MeasureResult]
]
AutoschedulerTuningRecordType = typing.List[
    typing.Tuple[auto_scheduler.MeasureInput, auto_scheduler.MeasureResult]
]
MetaScheduleTuningRecordType = typing.List[meta_schedule.database.TuningRecord]
TuningRecordsType = typing.Union[
    AutoTVMTuningRecordType,
    AutoschedulerTuningRecordType,
    MetaScheduleTuningRecordType,
]

# TODO(aluo): look into having autoscheduler handle file like objects instead of strings,
# Use StringIO handles instead of writing to files


def _infer_encoded_record_type(records: EncodedTuningRecordsType) -> RecordType:
    """Infers the record type (AutoTVM/Autoscheduler/MetaSchedule) of serialized tuning records.

    :param records: a list of serialized tuning records
    :return: the record type
    """
    if len(records) > 0:
        first_record = json.loads(records[0])
        if "input" in first_record:
            return RecordType.AUTOTVM
        elif "i" in first_record:
            return RecordType.AUTOSCHEDULE
        elif "workload" in first_record:
            return RecordType.METASCHEDULE
        else:
            raise TuningRecordFormatError(
                "Records must be in either AutoTVM or Autoscheduler formats."
            )
    else:
        # Treat empty records as AutoTVM.
        return RecordType.AUTOTVM


def infer_record_type(records: TuningRecordsType) -> RecordType:
    """Infers the record type (AutoTVM/Autoscheduler/MetaSchedule) of tuning records.

    :param records: a list of tuning records
    :return: the record type
    """
    if records and isinstance(records[0], meta_schedule.database.TuningRecord):
        return RecordType.METASCHEDULE
    elif len(records) == 0 or isinstance(records[0][0], autotvm.MeasureInput):
        return RecordType.AUTOTVM
    elif isinstance(records[0][0], auto_scheduler.MeasureInput):
        return RecordType.AUTOSCHEDULE
    else:
        raise TuningRecordFormatError(
            "Records must be either AutoTVM or Autoscheduler types."
        )


def decode_records(tuning_records: EncodedTuningRecordsType) -> TuningRecordsType:
    """Deserializes a list of serialized records.

    :param tuning_records: a list of serialized AutoTVM, AutoScheduler or
        MetaSchedule records.
    :return: the deserialized records
    """
    record_type = _infer_encoded_record_type(tuning_records)
    if record_type == RecordType.AUTOTVM:
        return list(map(autotvm.record.decode, tuning_records))
    elif record_type == RecordType.AUTOSCHEDULE:
        # Note load_record_from_string returns a list, not a tuple like it says.
        return list(
            map(
                tuple,
                map(
                    auto_scheduler.measure_record.load_record_from_string,
                    tuning_records,
                ),
            )
        )
    elif record_type == RecordType.METASCHEDULE:
        workloads: typing.Dict[str, typing.Any] = {}
        records: typing.List[meta_schedule.database.TuningRecord] = []

        for record in tuning_records:
            record_json = json.loads(record)
            if "workload" in record_json:
                workloads[
                    record_json["workload"][0]
                ] = meta_schedule.database.Workload.from_json(record_json["workload"])
            else:
                records.append(
                    meta_schedule.database.TuningRecord.from_json(
                        record_json["record"], workloads[record_json["workload_hash"]]
                    )
                )
        return records
    else:
        raise TuningRecordTypeError(f"Unknown TuningRecordType {record_type}")


def encode_records(tuning_records: TuningRecordsType) -> EncodedTuningRecordsType:
    """Encodes a list of records using AutoTVM/Autoscheduler/MetaSchedule
    serializers.

    :param tuning_records: List of records to encode
    :returns: List of json encoded records
    """
    record_type = infer_record_type(tuning_records)
    if record_type == RecordType.AUTOTVM:
        return [
            autotvm.record.encode(measure_input, measure_result)
            for measure_input, measure_result in tuning_records
        ]
    elif record_type == RecordType.AUTOSCHEDULE:
        return [
            auto_scheduler.measure_record.dump_record_to_string(
                measure_input, measure_result
            )
            for measure_input, measure_result in tuning_records
        ]
    else:
        raise TuningRecordTypeError(f"Unknown TuningRecordType {record_type}")


def read_tuning_records(path: pathlib.Path) -> TuningRecordsType:
    """Deserializes tuning records from a file.

    :param path: the path of the file to read
    :return: deserialized tuning records
    """
    # N.B. We rely on the fact that both AutoTVM, Autoscheduler and MetaSchedule
    # formats put one record on each line.
    with open(path, "r") as f:
        lines = f.readlines()
        return decode_records(lines)


def write_tuning_records(path: pathlib.Path, records: TuningRecordsType) -> None:
    """Serializes tuning records to a file.

    :param path: the path of the file to write to
    """
    str_path = str(path)
    record_type = infer_record_type(records)
    if record_type == RecordType.AUTOTVM:
        with open(str_path, "a") as f:
            for encoded_record in encode_records(records):
                f.write(str(encoded_record) + "\n")
    elif record_type == RecordType.AUTOSCHEDULE:
        return auto_scheduler.save_records(
            str_path,
            [record[0] for record in records],
            [record[1] for record in records],
        )
    elif record_type == RecordType.METASCHEDULE:
        with open(str_path, "a") as f:
            for encoded_record in encode_records(records):
                f.write(f"{encoded_record}\n")
    else:
        raise TuningRecordTypeError(f"Unknown TuningRecordType {record_type}")


# Support for segmented tuning record files


class TuningRecordKind(enum.Enum):
    """A kind of tuning record: best versus full log records."""

    BEST_LOG_RECORDS = "BEST_LOG_{tid}"
    FULL_LOG_RECORDS = "FULL_LOG_{tid}"

    def format(self, tid: int) -> str:
        return self.value.format(tid=tid)

    def is_match(self, path: pathlib.Path) -> bool:
        return bool(parse.parse(self.value, path.name))


def write_segmented_tuning_records(
    kind: TuningRecordKind, work_dir: pathlib.Path, tid: int, records: TuningRecordsType
) -> None:
    """Serializes tuning records to a file.

    :param kind: The kind of tuning record
    :param work_dir: the path of the directory to write to.
    :param tid: the unique task ID for this batch of records.
    :param records: The list of tuning records to record.
    """
    pth = work_dir / kind.format(tid)
    write_tuning_records(pth, records)


def get_segmented_tuning_record_files(
    kind: TuningRecordKind, work_dir: pathlib.Path
) -> typing.Iterable[pathlib.Path]:
    """Returns the set of files containing tuning records of the given kind."""
    return (phile for phile in work_dir.iterdir() if kind.is_match(phile))


def coalesce_segmented_tuning_records_to_file(
    kind: TuningRecordKind, work_dir: pathlib.Path, out_file: pathlib.Path
) -> None:
    """Write back tuning records to a single file.

    :param kind: The kind of tuning record
    :param work_dir: the path of the directory to write to.
    :param out_file: The output file
    """
    with open(out_file, "w+b") as fh:
        for phile in get_segmented_tuning_record_files(kind, work_dir):
            buf = phile.read_bytes()
            fh.write(buf)


def read_segmented_tuning_records(
    kind: TuningRecordKind, work_dir: pathlib.Path
) -> typing.Iterable[TuningRecordsType]:
    """Read segmented tuning records from a directory.

    :param kind: The kind of tuning records to retrieve.
    :param work_dir: The input directory
    :return: An iterator over tuning records
    """
    for phile in get_segmented_tuning_record_files(kind, work_dir):
        yield from read_tuning_records(phile)


def read_segmented_tuning_bytes(
    kind: TuningRecordKind, work_dir: pathlib.Path
) -> bytes:
    """Read segmented tuning record bytes from a directory.

    :param kind: The kind of tuning records to retrieve.
    :param work_dir: The input directory
    :return: A byte buffer containing tuning record data
    """
    return b"".join(
        phile.read_bytes()
        for phile in get_segmented_tuning_record_files(kind, work_dir)
    )
